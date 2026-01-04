import numpy as np
import matplotlib.pyplot as plt

class ABCAlgorithm:
    def __init__(self, obj_func, bounds, pop_size=30, max_iter=100, limit=20):
        """
        Yapay Arı Kolonisi (ABC) Algoritması
        :param obj_func: Optimize edilecek (minimize edilecek) amaç fonksiyonu
        :param bounds: Her boyut için (min, max) sınırlarını içeren liste
        :param pop_size: Besin kaynağı sayısı (Toplam popülasyonun yarısı)
        :param max_iter: Maksimum iterasyon sayısı
        :param limit: Gözcü arılar için deneme sınırı (iyileşmeyen çözümün terk edilmesi için)
        """
        self.obj_func = obj_func
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.limit = limit
        
        # Besin kaynaklarını rastgele başlat (Arama uzayı sınırları içinde)
        self.foods = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (self.pop_size, self.dim))
        self.fitness = np.zeros(self.pop_size)
        self.trials = np.zeros(self.pop_size)
        
        # Başlangıç değerlendirmesi: Her besin kaynağının uygunluk değerini hesapla
        for i in range(self.pop_size):
            self.fitness[i] = self.calculate_fitness(self.obj_func(self.foods[i]))
            
        # En iyi çözümü kaydet
        self.best_food = self.foods[np.argmax(self.fitness)].copy()
        self.best_fitness = np.max(self.fitness)
        self.best_score = self.obj_func(self.best_food)
        
        # Gelişimi takip etmek için geçmiş listesi
        self.history = []

    def calculate_fitness(self, f_val):
        """Uygunluk (Fitness) fonksiyonu: Amaç fonksiyonu değerini pozitif bir uygunluk değerine dönüştürür"""
        if f_val >= 0:
            return 1 / (1 + f_val)
        else:
            return 1 + abs(f_val)

    def employed_bees(self):
        """Görevli Arılar Aşaması: Her besin kaynağı için komşuluk araştırması yapılır"""
        for i in range(self.pop_size):
            # Rastgele bir komşu besin kaynağı ve boyut seç
            k = np.random.randint(0, self.pop_size)
            while k == i:
                k = np.random.randint(0, self.pop_size)
            
            j = np.random.randint(0, self.dim)
            phi = np.random.uniform(-1, 1) # Değişim hızı/rastgelelik katsayısı
            
            # Yeni bir aday çözüm üret (v_i = x_i + phi * (x_i - x_k))
            v_i = self.foods[i].copy()
            v_i[j] = self.foods[i, j] + phi * (self.foods[i, j] - self.foods[k, j])
            
            # Sınır kontrolü yap
            v_i[j] = np.clip(v_i[j], self.bounds[j, 0], self.bounds[j, 1])
            
            # Yeni adayı değerlendir
            f_new = self.obj_func(v_i)
            fit_new = self.calculate_fitness(f_new)
            
            # Eğer yeni çözüm daha iyiyse güncelle, değilse deneme sayısını artır
            if fit_new > self.fitness[i]:
                self.foods[i] = v_i
                self.fitness[i] = fit_new
                self.trials[i] = 0
            else:
                self.trials[i] += 1

    def onlooker_bees(self):
        """Gözcü Arılar Aşaması: Besin kaynakları uygunluklarına göre olasılıksal olarak seçilir ve aranır"""
        # Uygunluk değerlerine dayalı seçim olasılıklarını hesapla (Rulet tekerleği seçimi)
        probs = self.fitness / np.sum(self.fitness)
        
        count = 0
        i = 0
        while count < self.pop_size:
            # Olasılığa göre arı bir besin kaynağını seçer
            if np.random.rand() < probs[i]:
                # Görevli arılarda olduğu gibi komşuluk araması yap
                k = np.random.randint(0, self.pop_size)
                while k == i:
                    k = np.random.randint(0, self.pop_size)
                
                j = np.random.randint(0, self.dim)
                phi = np.random.uniform(-1, 1)
                
                v_i = self.foods[i].copy()
                v_i[j] = self.foods[i, j] + phi * (self.foods[i, j] - self.foods[k, j])
                v_i[j] = np.clip(v_i[j], self.bounds[j, 0], self.bounds[j, 1])
                
                f_new = self.obj_func(v_i)
                fit_new = self.calculate_fitness(f_new)
                
                if fit_new > self.fitness[i]:
                    self.foods[i] = v_i
                    self.fitness[i] = fit_new
                    self.trials[i] = 0
                else:
                    self.trials[i] += 1
                
                count += 1
            i = (i + 1) % self.pop_size

    def scout_bees(self):
        """Kaşif Arılar Aşaması: Belirli bir limit boyunca iyileşmeyen besin kaynakları terk edilir"""
        for i in range(self.pop_size):
            if self.trials[i] > self.limit:
                # Besin kaynağını terk et ve rastgele yeni bir tane bul
                self.foods[i] = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], self.dim)
                self.fitness[i] = self.calculate_fitness(self.obj_func(self.foods[i]))
                self.trials[i] = 0

    def solve(self):
        """Ana Optimizasyon Döngüsü"""
        for it in range(self.max_iter):
            # 1. Aşama: Görevli Arılar
            self.employed_bees()
            
            # 2. Aşama: Gözcü Arılar
            self.onlooker_bees()
            
            # En iyi çözümü güncelle
            current_best_idx = np.argmax(self.fitness)
            if self.fitness[current_best_idx] > self.best_fitness:
                self.best_fitness = self.fitness[current_best_idx]
                self.best_food = self.foods[current_best_idx].copy()
                self.best_score = self.obj_func(self.best_food)
            
            # 3. Aşama: Kaşif Arılar
            self.scout_bees()
            
            # İlerlemeyi kaydet
            self.history.append(self.best_score)
            
            if (it + 1) % 10 == 0:
                print(f"İterasyon {it+1}: En İyi Skor = {self.best_score:.20f}")
        
        return self.best_food, self.best_score

def sphere_function(x):
    """Test için küre fonksiyonu (Minimum: f(0,0...)=0)"""
    return np.sum(x**2)

def rosenbrock_function(x):
    """Rosenbrock fonksiyonu (Minimum: f(1,1...)=0)"""
    return np.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

def rastrigin_function(x):
    """Rastrigin fonksiyonu (Minimum: f(0,0...)=0)"""
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

if __name__ == "__main__":
    # Sonuçları daha hassas göstermek için numpy ayarı
    np.set_printoptions(precision=20, suppress=True)

    # Test edilecek fonksiyonlar ve parametreleri
    benchmarks = [
        {
            "name": "Sphere",
            "func": sphere_function,
            "bounds": [(-5.12, 5.12)] * 5,
            "pop_size": 50,
            "max_iter": 200,
            "limit": 30
        },
        {
            "name": "Rosenbrock",
            "func": rosenbrock_function,
            "bounds": [(-5, 10)] * 5,
            "pop_size": 50,
            "max_iter": 1000, # Rosenbrock daha zor olduğu için iterasyonu artırdım
            "limit": 50
        },
        {
            "name": "Rastrigin",
            "func": rastrigin_function,
            "bounds": [(-5.12, 5.12)] * 5,
            "pop_size": 50,
            "max_iter": 300,
            "limit": 50
        }
    ]

    for benchmark in benchmarks:
        print(f"\n--- {benchmark['name']} Fonksiyonu Testi Başlıyor ---")
        
        # Algoritmayı başlat
        abc = ABCAlgorithm(
            benchmark["func"], 
            benchmark["bounds"], 
            pop_size=benchmark["pop_size"], 
            max_iter=benchmark["max_iter"], 
            limit=benchmark["limit"]
        )
        best_sol, best_score = abc.solve()
        
        print(f"\nFinal Sonuçları ({benchmark['name']}):")
        print(f"En İyi Çözüm: {best_sol}")
        print(f"En İyi Skor: {best_score:.20f}")
        
        # Yakınsama Grafiği
        plt.figure(figsize=(10, 5))
        plt.plot(abc.history)
        plt.title(f"ABC Algoritması Yakınsama Grafiği ({benchmark['name']})")
        plt.xlabel("İterasyon")
        plt.ylabel("Amaç Fonksiyonu Değeri")
        plt.yscale('log')
        plt.grid(True)
        filename = f"convergence_plot_{benchmark['name'].lower()}.png"
        plt.savefig(filename)
        print(f"Yakınsama grafiği '{filename}' olarak kaydedildi.")
        plt.close() # Önceki grafiği kapat
        # plt.show() # Döngüde her seferinde durmaması için kapattım, istenirse açılabilir
