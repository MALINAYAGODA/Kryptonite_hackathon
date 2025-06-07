#!/usr/bin/env python3
"""
Система обучения нейронной сети для распознавания лиц с использованием адаптивного маржина.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from PIL import Image
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scripts.utils.utils import load_pretrained_model, load_checkpoint


performance_tracker = []
best_models_registry = []


def export_performance_log():
    """Экспортирует журнал производительности в JSON формат."""
    log_path = "training_logs/performance_history.json"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(performance_tracker, f, indent=2, ensure_ascii=False)
    
    wandb.save(log_path)
    print(f"История производительности сохранена в {log_path}")


class FacePairDataLoader(Dataset):
    """Загрузчик данных для пар лиц из JSON конфигурации."""

    def __init__(self, config_path, image_transforms):
        self.pairs_info = []
        self.transforms = image_transforms
        
        with open(config_path, "r", encoding="utf-8") as f:
            import csv
            reader = csv.reader(f)
            next(reader)
            for line in reader:
                self.pairs_info.append(line)

        valid_pairs = []
        for pair_data in self.pairs_info:
            first_img_path = pair_data[0]
            second_img_path = pair_data[1]
            if Path(first_img_path).exists() and Path(second_img_path).exists():
                valid_pairs.append(pair_data)
            else:
                print(f"Отсутствует файл: {first_img_path} или {second_img_path}")
        self.pairs_info = valid_pairs

    def __len__(self):
        return len(self.pairs_info)

    def __getitem__(self, index):
        pair_data = self.pairs_info[index]
        first_path = pair_data[0]
        second_path = pair_data[1]
        similarity_label = int(pair_data[2])
        
        first_synthetic = int(pair_data[3]) if len(pair_data) >= 5 else 0
        second_synthetic = int(pair_data[4]) if len(pair_data) >= 5 else 0

        first_image = Image.open(first_path).convert("RGB")
        second_image = Image.open(second_path).convert("RGB")
        
        if self.transforms:
            first_image = self.transforms(first_image)
            second_image = self.transforms(second_image)
            
        return (first_image, second_image, similarity_label, 
                first_path, second_path, first_synthetic, second_synthetic)


class DynamicMarginLossFunction(nn.Module):
    """Функция потерь с динамическим маржином для обучения эмбеддингов лиц."""

    def __init__(self, scale_factor=64, initial_margin=0.35, adaptation_rate=0.1):
        super(DynamicMarginLossFunction, self).__init__()
        self.scale_factor = scale_factor
        self.initial_margin = initial_margin
        self.adaptation_rate = adaptation_rate
        self.binary_cross_entropy = nn.BCEWithLogitsLoss()

    def forward(self, similarity_scores, target_labels):
        adaptive_margin = self.initial_margin + self.adaptation_rate * (1 - similarity_scores)
        
        scaled_logits = self.scale_factor * (similarity_scores - adaptive_margin * target_labels.float())
        
        loss_value = self.binary_cross_entropy(scaled_logits, target_labels.float())
        return loss_value


def calculate_equal_error_rate(prediction_scores, true_labels):
    """Вычисляет равную ошибку первого и второго рода (EER)."""
    false_positive_rate, true_positive_rate, _ = roc_curve(true_labels, prediction_scores)
    false_negative_rate = 1 - true_positive_rate
    
    eer_index = np.nanargmin(np.abs(false_positive_rate - false_negative_rate))
    equal_error_rate = (false_positive_rate[eer_index] + false_negative_rate[eer_index]) / 2
    return equal_error_rate


def run_model_validation(neural_network, data_loader, computing_device, decision_threshold=0.2):
    """Проводит валидацию модели на тестовых данных."""
    neural_network.eval()
    
    correct_predictions = 0
    total_samples = 0
    all_similarity_scores = []
    all_true_labels = []
    
    error_statistics = {
        "same_person_errors": 0,
        "different_real_errors": 0,
        "synthetic_involved_errors": 0
    }
    
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Валидация модели", leave=True, dynamic_ncols=True)
        
        for batch_data in progress_bar:
            (first_imgs, second_imgs, labels, first_paths, second_paths,
             first_synthetic_flags, second_synthetic_flags) = batch_data
            
            with torch.amp.autocast('cuda'):
                first_imgs = first_imgs.to(computing_device)
                second_imgs = second_imgs.to(computing_device)
                labels = labels.to(computing_device)
                
                first_embeddings, _ = neural_network(first_imgs)
                second_embeddings, _ = neural_network(second_imgs)
                
                first_embeddings = nn.functional.normalize(first_embeddings, p=2, dim=1)
                second_embeddings = nn.functional.normalize(second_embeddings, p=2, dim=1)
                
                cosine_similarities = torch.sum(first_embeddings * second_embeddings, dim=1)
            
            predictions = (cosine_similarities >= decision_threshold).long()
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            all_similarity_scores.extend(cosine_similarities.cpu().numpy().tolist())
            all_true_labels.extend(labels.cpu().numpy().tolist())
            
            for i in range(len(labels)):
                actual_label = labels[i].item()
                predicted_label = predictions[i].item()
                
                if actual_label != predicted_label:
                    if actual_label == 1:
                        error_statistics["same_person_errors"] += 1
                    else:
                        if first_synthetic_flags[i] == 0 and second_synthetic_flags[i] == 0:
                            error_statistics["different_real_errors"] += 1
                        else:
                            error_statistics["synthetic_involved_errors"] += 1

    final_accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    equal_error_rate = calculate_equal_error_rate(np.array(all_similarity_scores), 
                                                  np.array(all_true_labels))
    
    return final_accuracy, equal_error_rate, error_statistics


def create_model_checkpoint(model, optimizer, lr_scheduler, current_epoch, 
                          batch_number, validation_accuracy, error_rate):
    """Создает и сохраняет чекпоинт модели с обновлением реестра лучших моделей."""
    
    for existing_record in best_models_registry:
        if existing_record["batch_number"] == batch_number:
            return existing_record["checkpoint_path"]
    
    checkpoint_filename = (
        f"training_logs/checkpoints/model_ep{current_epoch}_"
        f"batch{batch_number}_acc{validation_accuracy:.4f}_eer{error_rate:.4f}.pth"
    )
    
    os.makedirs(os.path.dirname(checkpoint_filename), exist_ok=True)
    
    model_state = {
        "training_epoch": current_epoch,
        "batch_number": batch_number,
        "network_weights": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": lr_scheduler.state_dict(),
        "validation_accuracy": validation_accuracy,
        "equal_error_rate": error_rate,
    }
    torch.save(model_state, checkpoint_filename)

    performance_record = {
        "epoch": current_epoch,
        "batch_number": batch_number,
        "accuracy": validation_accuracy,
        "eer": error_rate,
        "checkpoint_path": checkpoint_filename
    }
    performance_tracker.append(performance_record)
    best_models_registry.append(performance_record)

    top_accuracy_models = sorted(best_models_registry, 
                               key=lambda x: x["accuracy"], reverse=True)[:5]
    top_eer_models = sorted(best_models_registry, 
                          key=lambda x: x["eer"])[:5]
    
    unique_models = {rec["batch_number"]: rec 
                    for rec in (top_accuracy_models + top_eer_models)}
    best_models_registry.clear()
    best_models_registry.extend(list(unique_models.values()))

    export_performance_log()
    print(f"Сохранен чекпоинт модели: {checkpoint_filename}")
    return checkpoint_filename


def execute_training_pipeline():
    """Основная функция запуска процесса обучения."""
    
    argument_parser = argparse.ArgumentParser(
        description="Обучение модели распознавания лиц с адаптивным маржином."
    )
    argument_parser.add_argument("--model_checkpoint", type=str,
                        default="training_logs/pretrained/ckpt_epoch1_batch43000_acc0.9825_eer0.0165.ckpt",
                        help="Путь к предобученной модели.")
    argument_parser.add_argument("--training_data", type=str,
                        default="datasets/training/face_pairs.csv",
                        help="CSV файл с тренировочными парами.")
    argument_parser.add_argument("--validation_data", type=str,
                        default="datasets/validation/face_pairs.csv",
                        help="CSV файл с валидационными парами.")
    argument_parser.add_argument("--training_epochs", type=int, default=2,
                        help="Количество эпох для обучения.")
    argument_parser.add_argument("--batch_size", type=int, default=32,
                        help="Размер батча.")
    argument_parser.add_argument("--learning_rate", type=float, default=5e-6,
                        help="Скорость обучения.")
    argument_parser.add_argument("--save_frequency", type=int, default=100,
                        help="Частота сохранения чекпоинтов (в батчах).")
    argument_parser.add_argument("--scale_parameter", type=float, default=64,
                        help="Параметр масштабирования для функции потерь.")
    argument_parser.add_argument("--margin_base", type=float, default=0.35,
                        help="Базовый маржин для функции потерь.")
    argument_parser.add_argument("--margin_adaptation", type=float, default=0.1,
                        help="Коэффициент адаптации маржина.")
    argument_parser.add_argument("--regularization", type=float, default=1e-2,
                        help="Коэффициент регуляризации.")
    argument_parser.add_argument("--model_architecture", type=str, default="ir_101",
                        help="Архитектура нейронной сети.")
    
    config = argument_parser.parse_args()

    wandb.init(
        project="face_recognition_training",
        config={
            "epochs": config.training_epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "save_frequency": config.save_frequency,
            "scale_parameter": config.scale_parameter,
            "margin_base": config.margin_base,
            "margin_adaptation": config.margin_adaptation,
            "regularization": config.regularization,
        }
    )

    computing_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    neural_network = load_pretrained_model(config.model_architecture).to(computing_device)
    neural_network = load_checkpoint(config.model_checkpoint, neural_network).to(computing_device)

    image_preprocessing = Compose([
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    training_dataset = FacePairDataLoader(config.training_data, image_transforms=image_preprocessing)
    validation_dataset = FacePairDataLoader(config.validation_data, image_transforms=image_preprocessing)
    
    training_loader = DataLoader(training_dataset, batch_size=config.batch_size,
                               shuffle=True, num_workers=8, pin_memory=True)
    validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size,
                                 shuffle=False, num_workers=8, pin_memory=True)

    loss_function = DynamicMarginLossFunction(
        scale_factor=config.scale_parameter,
        initial_margin=config.margin_base,
        adaptation_rate=config.margin_adaptation
    )
    
    optimizer = optim.AdamW(neural_network.parameters(), 
                          lr=config.learning_rate,
                          weight_decay=config.regularization)
    
    learning_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
        threshold=1e-4,
    )

    total_epochs = config.training_epochs
    checkpoint_frequency = config.save_frequency
    global_batch_index = -1

    gradient_scaler = torch.amp.GradScaler("cuda")

    for epoch_num in range(1, total_epochs + 1):
        neural_network.train()
        epoch_loss_accumulator = 0.0
        checkpoint_loss_accumulator = 0.0
        batches_since_checkpoint = 0

        epoch_progress = tqdm(training_loader,
                            desc=f"Эпоха {epoch_num}/{total_epochs} - Обучение",
                            leave=True, dynamic_ncols=True)

        for batch_data in epoch_progress:
            global_batch_index += 1
            batches_since_checkpoint += 1

            (first_images, second_images, similarity_labels, 
             first_paths, second_paths, _, _) = batch_data

            first_images = first_images.to(computing_device)
            second_images = second_images.to(computing_device)
            similarity_labels = similarity_labels.to(computing_device)

            optimizer.zero_grad()
            
            with torch.amp.autocast("cuda"):
                first_embeddings, _ = neural_network(first_images)
                second_embeddings, _ = neural_network(second_images)
                
                first_embeddings = nn.functional.normalize(first_embeddings, p=2, dim=1)
                second_embeddings = nn.functional.normalize(second_embeddings, p=2, dim=1)
                
                cosine_similarity = torch.sum(first_embeddings * second_embeddings, dim=1)
                
                batch_loss = loss_function(cosine_similarity, similarity_labels)
            
            gradient_scaler.scale(batch_loss).backward()
            gradient_scaler.step(optimizer)
            gradient_scaler.update()

            batch_size = first_images.size(0)
            epoch_loss_accumulator += batch_loss.item() * batch_size
            checkpoint_loss_accumulator += batch_loss.item() * batch_size

            if global_batch_index % checkpoint_frequency == 0:
                avg_checkpoint_loss = checkpoint_loss_accumulator / (batches_since_checkpoint * config.batch_size)
                
                validation_accuracy, eer_score, error_breakdown = run_model_validation(
                    neural_network, validation_loader, computing_device, threshold=0.2
                )
                
                metrics_log = {
                    "global_batch": global_batch_index,
                    "checkpoint_loss": avg_checkpoint_loss,
                    "validation_accuracy": validation_accuracy,
                    "equal_error_rate": eer_score,
                    "same_person_errors": error_breakdown["same_person_errors"],
                    "different_real_errors": error_breakdown["different_real_errors"],
                    "synthetic_errors": error_breakdown["synthetic_involved_errors"]
                }
                print("\n", metrics_log, "\n")
                wandb.log(metrics_log)

                learning_scheduler.step(eer_score)

                should_save_checkpoint = False
                if len(best_models_registry) < 3:
                    should_save_checkpoint = True
                else:
                    min_accuracy = min(best_models_registry, key=lambda x: x["accuracy"])["accuracy"]
                    max_eer = max(best_models_registry, key=lambda x: x["eer"])["eer"]
                    if validation_accuracy > min_accuracy or eer_score < max_eer:
                        should_save_checkpoint = True
                
                if should_save_checkpoint:
                    create_model_checkpoint(neural_network, optimizer, learning_scheduler, 
                                          epoch_num, global_batch_index, 
                                          validation_accuracy, eer_score)

                checkpoint_loss_accumulator = 0.0
                batches_since_checkpoint = 0

        average_epoch_loss = epoch_loss_accumulator / len(training_dataset)
        current_learning_rate = optimizer.param_groups[0]["lr"]
        
        epoch_summary = {
            "epoch": epoch_num,
            "average_loss": average_epoch_loss,
            "learning_rate": current_learning_rate
        }
        print(epoch_summary)
        wandb.log(epoch_summary)
        print(f"Эпоха {epoch_num} завершена, текущая скорость обучения: {current_learning_rate:.2e}")

    wandb.finish()


if __name__ == "__main__":
    execute_training_pipeline()
