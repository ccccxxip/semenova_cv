import cv2
import numpy as np
import pyautogui
import mss
import time
from skimage.measure import compare_ssim

class DinoGame:
    def __init__(self):
        self.sct = mss.mss()
        self.dino_region = {'top': 300, 'left': 100, 'width': 800, 'height': 200}
        self.jump_cooldown = 0
        self.ducking = False
        self.game_speed = 0
        self.score = 0
        self.last_score_check = time.time()
        
        # Загрузка шаблонов для сравнения
        self.obstacle_templates = {
            'cactus1': cv2.imread('cactus1.png', 0),
            'cactus2': cv2.imread('cactus2.png', 0),
            'bird': cv2.imread('bird.png', 0)
        }
        
        # Кадрирование области игры
        self.game_area = {'top': 300, 'left': 100, 'width': 800, 'height': 200}
        
    def get_screenshot(self):
        return np.array(self.sct.grab(self.dino_region))
    
    def preprocess_image(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return threshold
    
    def detect_obstacles(self, processed_img):
        obstacles = []
        
        # Проверка наличия препятствий в правой части экрана
        right_third = processed_img[:, -processed_img.shape[1]//3:]
        
        # Если есть черные пиксели (препятствия) в правой трети
        if np.any(right_third < 255):
            # Определяем положение нижнего препятствия (кактус)
            bottom_pixels = processed_img[-50:, -processed_img.shape[1]//3:]
            if np.any(bottom_pixels < 255):
                obstacles.append(('cactus', np.argwhere(bottom_pixels < 255)[0][1]))
            
            # Определяем положение верхнего препятствия (птица)
            top_pixels = processed_img[:100, -processed_img.shape[1]//3:]
            if np.any(top_pixels < 255):
                obstacles.append(('bird', np.argwhere(top_pixels < 255)[0][1]))
        
        return obstacles
    
    def take_action(self, obstacles):
        current_time = time.time()
        
        # Проверка и обновление счета
        if current_time - self.last_score_check > 1:
            self.update_score()
            self.last_score_check = current_time
        
        # Охлаждение после прыжка
        if self.jump_cooldown > 0:
            self.jump_cooldown -= 1
            return
        
        for obstacle, distance in obstacles:
            if obstacle == 'cactus' and distance < 100:
                # Прыжок для кактуса
                pyautogui.keyDown('space')
                time.sleep(0.05)
                pyautogui.keyUp('space')
                self.jump_cooldown = 10
                break
            elif obstacle == 'bird' and distance < 150:
                # Пригибание для птицы
                if not self.ducking:
                    pyautogui.keyDown('down')
                    self.ducking = True
                    time.sleep(0.3)
                else:
                    pyautogui.keyUp('down')
                    self.ducking = False
                break
    
    def update_score(self):
        # Здесь можно добавить OCR для чтения счета, но для простоты будем считать время
        self.score += 100 * self.game_speed
        print(f"Current score: {self.score}")
        
        # Увеличиваем сложность со временем
        if self.score > 5000:
            self.game_speed = 2
        elif self.score > 3000:
            self.game_speed = 1.5
        elif self.score > 1000:
            self.game_speed = 1
    
    def run(self):
        print("Starting Dino game bot in 3 seconds...")
        time.sleep(3)
        
        try:
            while True:
                screenshot = self.get_screenshot()
                processed_img = self.preprocess_image(screenshot)
                obstacles = self.detect_obstacles(processed_img)
                
                if obstacles:
                    self.take_action(obstacles)
                
                # Отладочная информация (можно закомментировать)
                debug_img = processed_img.copy()
                cv2.putText(debug_img, f"Score: {self.score}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
                cv2.imshow('Debug', debug_img)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Замедление игры для более стабильной работы
                time.sleep(0.02 + (0.01 * self.game_speed))
                
                # Остановка после 10000 очков
                if self.score >= 10000:
                    print("Reached 10000 points! Stopping...")
                    break
                    
        finally:
            cv2.destroyAllWindows()
            if self.ducking:
                pyautogui.keyUp('down')

if __name__ == "__main__":
    dino_bot = DinoGame()
    dino_bot.run()