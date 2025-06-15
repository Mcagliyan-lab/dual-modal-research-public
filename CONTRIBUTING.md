# 🤝 Katkıda Bulunma Rehberi

Dual-Modal Neural Network Research Projesi'ne katkıda bulunmakla ilgilendiğiniz için teşekkür ederiz! Bu proje, açık kaynak topluluğunun gücüne inanmaktadır ve her türlü katkıyı memnuniyetle karşılamaktadır. Katkılarınız, projenin gelişiminde kritik bir rol oynamaktadır.

Lütfen katkıda bulunmadan önce bu rehberi dikkatlice okuyunuz. Bu rehber, kod katkıları, hata raporları, özellik istekleri ve dokümantasyon güncellemeleri için yönergeler sağlar.

---

## 📝 İçindekiler

1.  [Davranış Kuralları](#1-davranış-kuralları)
2.  [Nasıl Katkıda Bulunulur?](#2-nasıl-katkıda-bulunulur)
    *   [Hata Raporları](#hata-raporları)
    *   [Özellik İstekleri](#özellik-istekleri)
    *   [Kod Katkıları](#kod-katkıları)
    *   [Dokümantasyon Katkıları](#dokümantasyon-katkıları)
3.  [Geliştirme Ortamı Kurulumu](#3-geliştirme-ortamı-kurulumu)
4.  [Kodlama Standartları](#4-kodlama-standartları)
5.  [Lisanslama](#5-lisanslama)

---

## 1. Davranış Kuralları

Bu projeye katılan herkesin topluluğumuza olumlu ve kapsayıcı bir deneyim sunmak için [Code of Conduct](CODE_OF_CONDUCT.md) dosyamızda belirtilen davranış kurallarına uyması beklenmektedir.

---

## 2. Nasıl Katkıda Bulunulur?

### Hata Raporları

Bir hata bulduysanız, lütfen GitHub Issues üzerinden bir hata raporu oluşturun. Hata raporunuzda aşağıdaki bilgileri sağladığınızdan emin olun:

*   Hatayı nasıl yeniden oluşturabileceğinize dair açık ve detaylı adımlar.
*   Beklenen davranış ve gözlemlenen davranış.
*   Kullandığınız işletim sistemi, Python sürümü ve kütüphane versiyonları.
*   Varsa ekran görüntüleri veya hata çıktıları.

### Özellik İstekleri

Yeni bir özellik veya geliştirme önermek istiyorsanız, lütfen GitHub Issues üzerinden bir özellik isteği oluşturun. İsteklerinizde aşağıdaki bilgileri ekleyin:

*   Özelliğin ne olduğunu ve neyi çözmeyi amaçladığını açıkça belirtin.
*   Neden bu özelliğin önemli olduğunu açıklayın.
*   Varsa kullanım senaryoları veya örnekler sunun.

### Kod Katkıları

Kod katkıları için genel iş akışı şöyledir:

1.  Depoyu forklayın.
2.  Yeni bir dal oluşturun (`git checkout -b feature/your-feature-name`).
3.  Değişikliklerinizi yapın ve testleri çalıştırın. Testleri çalıştırmak için proje kök dizininde `pytest` komutunu kullanabilirsiniz.
4.  Kodunuzun projenin [Kodlama Standartları](#4-kodlama-standartları)'na uygun olduğundan emin olun.
5.  Değişikliklerinizi commit edin (`git commit -m "feat: Add new feature"`). Açık ve açıklayıcı commit mesajları kullanın.
6.  Değişikliklerinizi forkladığınız depoya push edin (`git push origin feature/your-feature-name`).
7.  Orijinal depoya bir Pull Request (Çekme İsteği) açın. Pull Request'inizde şunları belirtin:
    *   Yaptığınız değişikliklerin kısa bir özeti.
    *   Çözdüğü sorun veya eklediği özellik.
    *   Nasıl test edildiğine dair bilgiler.

### Dokümantasyon Katkıları

Dokümantasyon güncellemeleri ve düzeltmeleri büyük beğeniyle karşılanır. Belgeleri daha anlaşılır, doğru ve kapsamlı hale getirmeye yardımcı olun. Kod katkılarına benzer şekilde, dokümantasyon değişiklikleri için de Pull Request süreci izlenir.

---

## 3. Geliştirme Ortamı Kurulumu

Projeyi yerel makinenizde kurmak için [Başlangıç Kılavuzu](docs/getting-started.md) bölümüne başvurun.

---

## 4. Kodlama Standartları

*   **Python:** PEP 8 standartlarına uyun. `flake8` veya `black` gibi formatlayıcılar kullanılması önerilir.
*   **Docstrings:** Tüm fonksiyonlar, sınıflar ve önemli metodlar için açıklayıcı docstringler (örneğin, Google tarzı) sağlayın.
*   **Testler:** Eklediğiniz veya değiştirdiğiniz her özellik için ilgili unit veya entegrasyon testlerini yazın. Çekme istekleriniz otomatik CI/CD testleri ile denetlenecektir.

---

## 5. Lisanslama

Bu projeye yaptığınız tüm katkılar, projenin ana lisansı olan [MIT Lisansı](../LICENSE) altında lisanslanacaktır. Katkıda bulunarak, kodunuzu bu koşullar altında lisanslamayı kabul etmiş olursunuz.

---

Teşekkür ederiz!

Dual-Modal Neural Network Research Projesi Ekibi 