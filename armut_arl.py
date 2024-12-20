# Association Rule Based Recommender System -- Tavsiye Sistemi

# Association Rule Learning : Veri içerisindeki örüntüleri bulmak için kullanılan
# kural tabanlı bir makine öğrenmesi yöntemidir.

# Armut Türkiyenin online en büyük hizmet platformu
# Armut hizmet verenler ile hizmet almak isteyenleri buluşturan bir platformdur.
# Veri Seti : Hizmet alan kullanıcılar ve bu kullanıcıarın almış oldukları servis ve kategorileri içerir.
# Bu veri setini kullanarak Association Rule Learning ile ürün tavsiye sistemi oluşturulmalıdır.

# Veri seti : müşterilerin aldıkları servisler, bu servislerin kategorileri yer alıyor.
# Her hizmetin tarih ve saat  bilgisi vardır.

# UserId : Müşteri Numarası
# ServiceId : Her kategoride anonimleştirilmiş servislerdir.






# 1- Veriyi hazırla
# 2- ServisId ve CategoryId _ ile birleştir yeni bir değişken oluştur.
# 3- Sepet oluşturmalıyız.Sepet ıd tanımla
# 4- Sepet, hizmet pivot table oluştur.
# 5- Birliktelik kurallarını oluştur.
# 6- arl_recommender fonksiyonu kullanarak en son 2_0 hizmetini alan bir kullanıcıya hizmet önerisinde bulununuz.


import pandas as pd
pd.set_option("display.max_columns", None)
from mlxtend.frequent_patterns import apriori, association_rules

# 1- Veriyi hazırla

armut = pd.read_csv("dataset/armut_data.csv")
df = armut.copy()
df.head()

# 2- ServisId ve CategoryId _ ile birleştir yeni bir değişken oluştur.

df["Hizmet"] = [str(row[1]) + "_" + str(row[2]) for row in df.values]
df.head()

# 3- Sepet oluşturmalıyız. Sepet ıd tanımla

# assciation rule learning uygulamak için sepet fatura tanımı oluşturmalıyız.

# sepet tanımı -- her müşterinin aylık aldığı hizmet
# her ay aldıkları farklı bir sepet olucak.
# Sepetleri unique bir Id ile tanımlanması gerekmektedir.
# Bunun için öncelikle sadece yıl ve ay içeren bir date değişkeni oluştur
# UserId ve yeni oluşturduğunuz date değişkenini kullanıcı bazında "_" ile
# birleştir ve ID adında yeni bir değişkene ata.


df.info()
df["CreateDate"] = pd.to_datetime(df["CreateDate"])
df["NewDate"] = df["CreateDate"].dt.strftime("%Y-%m")
df.head()

df["SepetId"] = [str(row[0]) + "_" + str(row[5]) for row in df.values]
df.head()

# 4- Sepet, hizmet pivot table oluştur.

invoice_product_df = df.groupby(["SepetId", "Hizmet"])["Hizmet"].count().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)
invoice_product_df.head()

# 5- Birliktelik kurallarını oluştur.

frequent_itemsets = apriori(invoice_product_df, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.head()


# 6- arl_recommender fonksiyonu kullanarak en son 2_0 hizmetini alan bir kullanıcıya hizmet önerisinde bulununuz.

def arl_recommender(rules_df, product_id, rec_count=1):

# rec_count = öneri sayısı
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list =[]
    for i, product in sorted_rules["antecedents"].items():
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))
    recommendation_list = list({item for item_list in recommendation_list for item in item_list})
    return recommendation_list[:rec_count]

arl_recommender(rules,"2_0",2)