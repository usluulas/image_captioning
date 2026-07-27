# image_captioning

Bir fotoğrafı yerel bir görsel-dil modeliyle (Ollama üzerinde `gemma3:4b-it-qat`) analiz edip
Türkçe tek cümlelik betimleme ve etiketler üreten terminal uygulaması.

---
## Docker olmadan çalıştırma

1. [Ollama](https://ollama.com/download)'yı kurun.
2. Modeli indirin: `ollama pull gemma3:4b-it-qat`
3. Bağımlılıkları kurun: `pip install -r requirements.txt`
4. Çalıştırın: `python image_captioning.py`

---

## Dosyalar

| Dosya | Görevi |
|---|---|
| `image_captioning.py` | Uygulamanın tamamı: prompt, görsel hazırlama, model çağrısı, terminal arayüzü |
| `requirements.txt` | Python bağımlılıkları (sürümler sabitlenmiştir) |
| `Dockerfile` | Python ortamının tarifi |
| `entrypoint.sh` | Container açılışında Ollama'yı bekler, model yoksa indirir |
| `docker-compose.yml` | Ollama sunucusu + uygulama container'ını birlikte ayağa kaldırır |
| `docker-compose.gpu.yml` | NVIDIA GPU kullanımı için ek ayar dosyası |
| `docker-compose.mac.yml` | macOS'ta Mac'e kurulu Ollama'ya bağlanan alternatif yapılandırma |

## Docker ile çalıştırma (önerilen — hiçbir şey kurmanız gerekmez)

### 1. Docker Desktop'ı kurun

- <https://www.docker.com/products/docker-desktop/> adresinden **Docker Desktop for Windows**'u indirin.
- Kurulumda **"Use WSL 2 instead of Hyper-V"** seçeneği işaretli kalsın.
- Kurulum bittikten sonra **bilgisayarı yeniden başlatın**.
- Docker Desktop'ı açın ve sol alttaki durum göstergesi yeşil **"Engine running"** olana kadar bekleyin.

Kontrol için terminalde:

```bash
docker --version
```

### 2. Projeyi çalıştırın

Proje klasöründe terminal açıp:

```bash
docker compose run --rm betimleyici
```

İlk çalıştırma uzun sürer: Docker imajı hazırlanır ve model (~4 GB) indirilir. Sonraki
çalıştırmalar saniyeler içinde başlar — model kalıcı bir Docker volume'ünde saklanır.

### 3. Fotoğrafı verin

Betimlenecek fotoğrafları proje içindeki **`gorseller/`** klasörüne kopyalayın.
Program yol sorduğunda `/gorseller/` ön ekiyle yazın:

```
Lütfen betimlenecek fotoğrafın yolunu girin:
> /gorseller/kelebek.jpg
```

Desteklenen formatlar: `.jpg .jpeg .png .webp .bmp .heic .heif`

### 4. (İsteğe bağlı) NVIDIA ekran kartı varsa

GPU kullanmak analizi 10-20 kat hızlandırır. NVIDIA kartı olan makinelerde:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm betimleyici
```

Ekran kartı yoksa bu komutu **kullanmayın**; normal komut CPU üzerinde çalışır
(fotoğraf başına yaklaşık 1-3 dakika, sonuç aynıdır).

### 5. (İsteğe bağlı) Mac kullanıyorsanız

Docker, macOS'ta Apple Silicon'ın GPU'suna (Metal) erişemez — normal komut Mac'te de yalnızca
CPU kullanır ve yavaş çalışır. Tam hız için Ollama'yı Mac'e normal uygulama olarak kurup
container'ı ona bağlayabilirsiniz:

1. <https://ollama.com/download> adresinden Ollama'yı kurun ve çalıştırın.
2. Ollama'nın container'dan gelen bağlantıları kabul etmesi için terminalde bir kez:

   ```bash
   launchctl setenv OLLAMA_HOST "0.0.0.0:11434"
   ```

   Ardından Ollama uygulamasını kapatıp yeniden açın.
3. Programı şu komutla çalıştırın:

   ```bash
   docker compose -f docker-compose.mac.yml run --rm betimleyici
   ```

Bu yolda Ollama container'ı hiç başlatılmaz; model Mac'in GPU'sunda çalışır, Python tarafı
yine container içinde izole kalır. Kolaylık isteniyorsa normal komut da Mac'te çalışır,
sadece yavaştır.

### Kapatma

Program bittikten sonra arka planda kalan Ollama sunucusunu durdurmak için:

```bash
docker compose down
```

### Bilinen sınırlama: çıktı birebir aynı olmayabilir

`temperature=0.0` ayarı modeli deterministik yapar, ancak bu determinizm **aynı donanım ve aynı
çalıştırma ortamı** için geçerlidir. Aynı görsel Docker içinde (CPU) ve Docker dışında (GPU)
çalıştırıldığında betimleme anlamca aynı, kelimesi kelimesine farklı çıkabilir — çünkü hesaplama
farklı sayısal yollardan geçer. Örnek:

| Ortam | Çıktı |
|---|---|
| Docker (CPU) | "Beyaz bir Fender gitarın yanında, desenli bir kedi oturan bir odada durmaktadır." |
| Yerel (GPU) | "Beyaz bir Fender Telecaster gitarın yanında bembeyaz ve gri-siyah çizgili bir kedi oturmuş durumda, arka planda büyük bir tablo ve televizyon bulunuyor." |

Docker'ın çözdüğü şey **"hiç çalışmıyor"** problemidir; **"kelimesi kelimesine aynı çıktı"** garantisi
vermez.

---

