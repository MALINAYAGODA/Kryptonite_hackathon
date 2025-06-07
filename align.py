#!/usr/bin/env python3
"""
Система предобработки и выравнивания изображений лиц для подготовки датасета.
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
warnings.filterwarnings("ignore", category=UserWarning)

import torch
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
import torchvision.transforms.functional as TF
from huggingface_hub import hf_hub_download
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from aligner.wrapper import CVLFaceAlignmentModel, ModelConfig

processing_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
alignment_config = None
face_processor = None


def process_single_image(source_image_path):
    image_transforms = Compose([
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    source_image = Image.open(source_image_path).convert("RGB")
    preprocessed_tensor = image_transforms(source_image).unsqueeze(0).to(processing_device)
    
    alignment_result = face_processor(preprocessed_tensor)
    processed_tensor = alignment_result[0][0] 
    
    processed_tensor = (processed_tensor * 0.5 + 0.5).clamp(0, 1)
    final_image = TF.to_pil_image(processed_tensor)
    
    return final_image


def batch_process_dataset(source_directory, destination_directory):
    """
    Обрабатывает весь датасет изображений, применяя выравнивание лиц.
    Поддерживает иерархическую структуру папок с изображениями.

    Args:
        source_directory (str): Исходная директория с необработанными изображениями.
        destination_directory (str): Целевая директория для сохранения обработанных изображений.
    """
    Path(destination_directory).mkdir(parents=True, exist_ok=True)
    
    subdirectories = sorted([d for d in os.listdir(source_directory) 
                           if os.path.isdir(os.path.join(source_directory, d))])
    
    processing_progress = tqdm(subdirectories, desc="Обработка датасета изображений")
    
    for subdir_name in processing_progress:
        source_subdir = os.path.join(source_directory, subdir_name)
        target_subdir = os.path.join(destination_directory, subdir_name)

        Path(target_subdir).mkdir(parents=True, exist_ok=True)
        
        image_files = [f for f in os.listdir(source_subdir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        for image_filename in image_files:
            source_image_path = os.path.join(source_subdir, image_filename)
            target_image_path = os.path.join(target_subdir, image_filename)
            
            try:
                aligned_face = process_single_image(source_image_path)
                
                aligned_face.save(target_image_path, quality=95, optimize=True)
                
            except Exception as processing_error:
                print(f"Ошибка при обработке {source_image_path}: {processing_error}")
                continue


def initialize_alignment_system():
    """Инициализирует систему выравнивания лиц."""
    global face_processor, alignment_config
    
    alignment_config = ModelConfig()
    
    face_processor = CVLFaceAlignmentModel(alignment_config).to(processing_device)
    face_processor.eval()
    
    print(f"Система выравнивания инициализирована на устройстве: {processing_device}")


def run_alignment_pipeline():
    """Основная функция запуска процесса выравнивания датасета."""
    
    cmd_parser = argparse.ArgumentParser(
        description="Система предобработки и выравнивания изображений лиц для датасета."
    )
    cmd_parser.add_argument("--source_path", type=str,
                        default="datasets/raw_images",
                        help="Путь к исходной директории с необработанными изображениями")
    cmd_parser.add_argument("--target_path", type=str,
                        default="datasets/processed_images",
                        help="Путь для сохранения обработанных и выровненных изображений")
    cmd_parser.add_argument("--batch_processing", action="store_true",
                        help="Включить пакетную обработку для больших датасетов")
    
    arguments = cmd_parser.parse_args()

    initialize_alignment_system()

    print(f"Начинается обработка датасета:")
    print(f"  Исходная директория: {arguments.source_path}")
    print(f"  Целевая директория: {arguments.target_path}")
    print(f"  Пакетная обработка: {'Включена' if arguments.batch_processing else 'Отключена'}")

    if not os.path.exists(arguments.source_path):
        print(f"Ошибка: Исходная директория {arguments.source_path} не найдена!")
        return
    
    try:
        batch_process_dataset(arguments.source_path, arguments.target_path)
        print(f"\nОбработка завершена успешно!")
        print(f"Обработанные изображения сохранены в: {arguments.target_path}")
        
    except Exception as main_error:
        print(f"Критическая ошибка при обработке датасета: {main_error}")
        return

    try:
        source_count = sum([len(files) for r, d, files in os.walk(arguments.source_path)])
        target_count = sum([len(files) for r, d, files in os.walk(arguments.target_path)])
        
        print(f"\nСтатистика обработки:")
        print(f"  Исходных изображений: {source_count}")
        print(f"  Обработанных изображений: {target_count}")
        print(f"  Успешность обработки: {(target_count/source_count*100):.1f}%" if source_count > 0 else "  Успешность обработки: 0%")
        
    except Exception as stats_error:
        print(f"Не удалось собрать статистику: {stats_error}")


if __name__ == "__main__":
    run_alignment_pipeline()
