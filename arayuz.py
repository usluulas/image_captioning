"""Görsel Betimleyici — grafik arayüz (Gradio).

Terminal sürümü (image_captioning_ui.py) olduğu gibi çalışmaya devam ediyor; bu dosya
sadece arayüz katmanı. Model çağrısı, prompt ve tüm ayarlar çekirdek dosyadan geliyor,
burada kopyalanmıyor — böylece prompt üzerinde yapılan iyileştirmeler iki sürümde de
otomatik geçerli oluyor.

Çalıştırmak için:  python arayuz.py      (tarayıcıda http://127.0.0.1:7860 açılır)
"""

import os
import threading
import time

# Gradio varsayılan olarak kendi sunucusuna kullanım istatistiği ve sürüm kontrolü gönderir.
# Bu satır import'tan ÖNCE gelmeli, yoksa kütüphane değeri okumadan kurulur. Program tamamen
# çevrimdışı çalışsın diye kapatıldı; internet olmayan makinede de sorunsuz açılır.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr

from image_captioning_ui import (
    DESTEKLENEN_UZANTILAR,
    MODEL_ADI,
    gorseli_betimle_yapisal,
)

# Docker içinde container dışından erişilebilmesi için 0.0.0.0 gerekir; docker-compose.yml
# bu değişkeni öyle veriyor. Docker dışında varsayılan 127.0.0.1 kalır, yani arayüz yerel
# ağa açılmaz — yalnızca bu bilgisayardan erişilir.
SUNUCU_ADRESI = os.environ.get("ARAYUZ_HOST", "127.0.0.1")
SUNUCU_PORTU = int(os.environ.get("ARAYUZ_PORT", "7860"))
YEREL_CALISMA = SUNUCU_ADRESI not in ("0.0.0.0", "::")

# İlerleme çubuğu için tahmini süre. Ollama tek bloklayıcı çağrı döndürüyor, gerçek yüzde
# bilgisi vermiyor; bu yüzden çubuk GEÇEN SÜREYE dayanıyor ve arayüzde "tahmini" diye
# belirtiliyor. Değer bu makinede ölçüldü (temiz koşulda 71-92 sn); ortam değişkeniyle
# değiştirilebilir, çünkü CPU-only makinelerde süre birkaç dakikayı bulabiliyor.
TAHMINI_SURE_SN = int(os.environ.get("TAHMINI_SURE_SN", "85"))

ACIKLAMA_METNI = f"""
Fotoğrafı aşağıdaki alana sürükleyin, ardından **Betimle** düğmesine basın.
Desteklenen formatlar: {', '.join(u.lstrip('.') for u in DESTEKLENEN_UZANTILAR)}.

Model: `{MODEL_ADI}` — görsel başına ekran kartıyla **~75-95 saniye**, ekran kartı
olmadan birkaç dakika sürer. Bu normaldir, sayfayı yenilemeyin.
"""


def betimle(dosya_yolu, progress=gr.Progress()):
    """Yüklenen görseli modele gönderir; (betimleme, etiketler, durum) döndürür."""
    if not dosya_yolu:
        return "", "", "Önce bir görsel yükleyin."

    uzanti = os.path.splitext(dosya_yolu)[1].lower()
    if uzanti not in DESTEKLENEN_UZANTILAR:
        return "", "", f"Desteklenmeyen dosya biçimi: {uzanti or 'bilinmiyor'}"

    # Model çağrısı bloklayıcı ve ara bilgi vermiyor; ayrı bir iş parçacığında çalıştırıp
    # burada geçen süreyi sayarak çubuğu ilerletiyoruz.
    kutu = {}

    def calistir():
        try:
            kutu["sonuc"] = gorseli_betimle_yapisal(dosya_yolu)
        except Exception as e:  # noqa: BLE001 - hata mesajı kullanıcıya gösterilecek
            kutu["istisna"] = e

    isci = threading.Thread(target=calistir, daemon=True)
    baslangic = time.time()
    isci.start()

    while isci.is_alive():
        gecen = int(time.time() - baslangic)
        if gecen < TAHMINI_SURE_SN:
            # Çubuk %99'da bekletiliyor; tahmin tutmazsa %100 gösterip yanıltmayalım.
            oran = min(gecen / TAHMINI_SURE_SN, 0.99)
            progress(oran, desc=f"Analiz ediliyor: {gecen} sn / ~{TAHMINI_SURE_SN} sn (tahmini)")
        else:
            progress(0.99, desc=f"Tahminden uzun sürüyor, devam ediyor: {gecen} sn")
        time.sleep(0.5)

    isci.join()
    gecen = round(time.time() - baslangic)

    if "istisna" in kutu:
        # En sık sebep: Ollama sunucusu çalışmıyor ya da model belleğe sığmadı.
        return "", "", (
            f"Bağlantı/çalışma hatası: {kutu['istisna']}\n\n"
            f"Ollama'nın açık olduğundan emin olun ('ollama run {MODEL_ADI}')."
        )

    sonuc = kutu["sonuc"]
    if sonuc["hata"]:
        return "", "", f"{sonuc['hata']}\n\nHam yanıt:\n{sonuc['ham_yanit']}"

    return sonuc["aciklama"], ", ".join(sonuc["etiketler"]), f"Betimleme hazır ({gecen} sn)."


# analytics_enabled=False: Gradio varsayılan olarak kullanım bilgisi gönderiyor, kapatıldı.
with gr.Blocks(title="Görsel Betimleyici", analytics_enabled=False) as arayuz:
    gr.Markdown("# Görsel Betimleyici\nArşiv için Türkçe betimleme ve etiket üretir.")
    gr.Markdown(ACIKLAMA_METNI)

    with gr.Row():
        with gr.Column():
            gorsel = gr.Image(
                label="Görsel",
                # "filepath": Gradio dosyayı yeniden kodlamadan olduğu gibi geçirir; böylece
                # EXIF döndürme etiketi korunur ve çekirdekteki düzeltme doğru çalışır.
                # (Yeniden kodlansaydı telefon fotoğrafları modele yan yatmış gidebilirdi.)
                type="filepath",
                sources=["upload", "clipboard"],
                height=360,
            )
            betimle_dugmesi = gr.Button("Betimle", variant="primary")

        with gr.Column():
            # buttons=["copy"]: arşiv sitesine yapıştırmak için tek tıkla panoya kopyalar.
            # (Gradio 5'teki show_copy_button parametresinin 6.x'teki karşılığı.)
            aciklama_kutusu = gr.Textbox(
                label="Betimleme",
                lines=4,
                buttons=["copy"],
                placeholder="Betimleme burada görünecek.",
            )
            etiket_kutusu = gr.Textbox(
                label="Etiketler",
                lines=2,
                buttons=["copy"],
                placeholder="Etiketler burada görünecek.",
            )
            durum = gr.Textbox(label="Durum", lines=2, interactive=False)

    # İki adım: önce eski sonuç temizlenir, sonra uzun süren model çağrısı başlar. İkinci
    # adımda ilerleme çubuğu geçen süreye göre dolar (bkz. betimle()).
    betimle_dugmesi.click(
        fn=lambda: ("", "", "Fotoğraf analiz ediliyor, lütfen bekleyin..."),
        outputs=[aciklama_kutusu, etiket_kutusu, durum],
    ).then(
        fn=betimle,
        inputs=gorsel,
        outputs=[aciklama_kutusu, etiket_kutusu, durum],
        show_progress="full",
    )


if __name__ == "__main__":
    # 4GB VRAM'de aynı anda iki çıkarım çalışırsa model belleğe sığmaz; istekler sıraya alınır.
    # api_open=False: arayüzün altındaki "API kullan" bağlantısı gizlenir, gerek yok.
    arayuz.queue(default_concurrency_limit=1, api_open=False)
    arayuz.launch(
        server_name=SUNUCU_ADRESI,
        server_port=SUNUCU_PORTU,
        inbrowser=YEREL_CALISMA,  # Docker'da tarayıcı yok, orada açmaya çalışma.
    )