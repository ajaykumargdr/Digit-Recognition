import cv2
import numpy as np
from pathlib import Path

def process_image(image_path):
    # Загрузка изображения
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot load image from {image_path}")
    
    # Инвертируем если фон светлый
    if np.mean(img) > 128:
        img = 255 - img
        
    # Удаление шума
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Бинаризация
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Найти контуры цифры
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Получить наибольший контур (предположительно цифра)
        main_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(main_contour)
        img = img[y:y+h, x:x+w]
    
    # Изменить размер до 28x28 с отступами
    target_size = 28
    ratio = float(target_size-4) / max(img.shape)
    resized = cv2.resize(img, None, fx=ratio, fy=ratio)
    
    # Создать пустое изображение 28x28
    output = np.zeros((target_size, target_size))
    
    # Центрировать изображение
    h, w = resized.shape
    offset_h = (target_size - h) // 2
    offset_w = (target_size - w) // 2
    output[offset_h:offset_h+h, offset_w:offset_w+w] = resized
    
    # Нормализация
    output = output / 255.0
    
    return output

def prepare_for_prediction(image_path):
    # Обработка изображения
    processed = process_image(image_path)
    
    # Подготовка для модели
    img_array = processed.reshape(1, 28, 28, 1)
    
    # Нормализация как в обучающем наборе
    mean = np.mean(img_array)
    std = np.std(img_array)
    img_array = (img_array - mean) / std
    
    return img_array
