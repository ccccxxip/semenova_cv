import socket
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

host = "84.237.21.36"
port = 5152

# данные с сеервера 
def recv_all(sock, n):
    data = bytearray()  
    while len(data) < n:  
        packet = sock.recv(n - len(data))  
        if not packet:  
            return None
        data.extend(packet)  
    return data  

# расстояния между двумя звездами
def calculate_distance(centroid1, centroid2):
    # формула расстояния между двумя точками
    return np.sqrt((centroid2[0] - centroid1[0])**2 + (centroid2[1] - centroid1[1])**2)

# поиск звезд на изображении
def find_stars(image, threshold=100):
    binary = image > threshold  # бинаризация 
    labeled = label(binary)  # маркировка
    regions = regionprops(labeled)  
    return regions  # возвращаем две точки

# отображение изображения и найденных звезд
def plot_stars(image, centroids=None):
    plt.clf()  # очистка
    plt.imshow(image)  
    if centroids:  
        for centroid in centroids:  
            # пометка звезды крестиком
            plt.plot(centroid[1], centroid[0], 'rx', markersize=7)
    plt.pause(0.7)  

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.connect((host, port))  
    response = b"" 
    
    # интерактивный режим отображения графиков
    plt.ion()
    plt.figure(figsize=(8, 6))  
    
    while response != b"yep":
        sock.send(b"get")  
        data = recv_all(sock, 40002) 
        if not data:
            break  
        
        # размеры изображения
        rows, cols = data[0], data[1]
        image = np.frombuffer(data[2:40002], dtype='uint8').reshape(rows, cols)
        
        # поиск звезд на изображении
        stars = find_stars(image)
        
        # проверка на минимум 2 звезды
        if len(stars) < 2:
            print("звезд не обнаружено или обнаружена всего 1 штука")
            continue  # пропуск если звезд недостаточно 
        
        # координаты центров двух самых ярких звезд
        star1, star2 = stars[0].centroid, stars[1].centroid
        # расстояние между ними
        distance = calculate_distance(star1, star2)
        
        # расстояние на сервер с точностью до 1 знака после запятой
        sock.send(f"{distance:.1f}".encode())
        # результаты
        print(f"коорд 1 звезды {star1}, коорд 2 зведы{star2}, расстояние между ними: {distance:.1f}")
        print("состояние", sock.recv(10))  
        
        # отображение 
        plot_stars(image, [star1, star2])
        
        sock.send(b"beat")

        response = sock.recv(10)
    
    print("ответ сервера", response.decode())