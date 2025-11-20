# NER_NEW_ATTEMPT

https://huggingface.co/knowledgator/gliner-x-large content : 

---
license: apache-2.0
language:
- multilingual
- da
- sv
- 'no'
- cs
- pl
- lt
- et
- lv
- es
- fi
- en
- de
- fr
- ro
- it
- pt
- nl
- ar
- zh
- hi
- uk
- sl
library_name: gliner
datasets:
- knowledgator/gliner-multilingual-synthetic
- lang-uk/UberText-NER-Silver
pipeline_tag: token-classification
tags:
- NER
- GLiNER
- information extraction
- encoder
- entity recognition
- multilingual
---
![alt text](image.png)

**GLiNER** is a Named Entity Recognition (NER) model capable of identifying any entity type using a bidirectional transformer encoders (BERT-like). It provides a practical alternative to traditional NER models, which are limited to predefined entities, and Large Language Models (LLMs) that, despite their flexibility, are costly and large for resource-constrained scenarios.

The initial GLiNER models were trained mainly on English data. Available multilingual models relied on existing multilingual NER datasets, but we prepared synthetical dataset [knowledgator/gliner-multilingual-synthetic](https://huggingface.co/datasets/knowledgator/gliner-multilingual-synthetic) using LLM to annotate [Fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) multilingual dataset. To enable broader language coverage, we replaced the `DeBERTa` backbone used in monolingual GLiNER models with `MT5` encoders, improving performance and adaptability across diverse languages.

Key Advantages Over Previous GLiNER Models:
* Enhanced performance and generalization capabilities
* Supports , `Swedish`, `Norwegian`, `Czech`, `Polish`, `Lithuanian`, `Estonian`, `Latvian`, `Spanish`, `Finnish`, `English`, `German`, `French`, `Romanian`, `Italian`, `Portuguese`, `Dutch`, `Ukrainian`, `Hindi`, `Chinese`, `Arabic`  languages.
* 3 model size available.

### Installation & Usage
Install or update the gliner package with all tokenizers that might be needed to process different languages:
```bash
pip install gliner[tokenizers] -U
```
Once you've downloaded the GLiNER library, you can import the GLiNER class. You can then load this model using `GLiNER.from_pretrained` and predict entities with `predict_entities`.

```python
from gliner import GLiNER
model = GLiNER.from_pretrained("knowledgator/gliner-x-large")
text = """
Cristiano Ronaldo dos Santos Aveiro (Portuguese pronunciation: [kɾiʃˈtjɐnu ʁɔˈnaldu]; born 5 February 1985) is a Portuguese professional footballer who plays as a forward for and captains both Saudi Pro League club Al Nassr and the Portugal national team. Widely regarded as one of the greatest players of all time, Ronaldo has won five Ballon d'Or awards,[note 3] a record three UEFA Men's Player of the Year Awards, and four European Golden Shoes, the most by a European player. He has won 33 trophies in his career, including seven league titles, five UEFA Champions Leagues, the UEFA European Championship and the UEFA Nations League. Ronaldo holds the records for most appearances (183), goals (140) and assists (42) in the Champions League, goals in the European Championship (14), international goals (128) and international appearances (205). He is one of the few players to have made over 1,200 professional career appearances, the most by an outfield player, and has scored over 850 official senior career goals for club and country, making him the top goalscorer of all time.
"""
labels = ["person", "award", "date", "competitions", "teams"]
entities = model.predict_entities(text, labels, threshold=0.5)
for entity in entities:
    print(entity["text"], "=>", entity["label"])
```

```
Cristiano Ronaldo dos Santos Aveiro => persona
5 de febrero de 1985 => fecha
Al Nassr de la Liga Profesional Saudí => equipos
selección nacional de Portugal => equipos
Balón de Oro => premio
Jugador del Año de la UEFA => premio
Botas de Oro europeas => premio
33 trofeos => premio
Ligas de Campeones de la UEFA => competiciones
Eurocopa => competiciones
Liga de Naciones de la UEFA => competiciones
Liga de Campeones => competiciones
```

<details>
<summary>Spanish</summary>

```python
from gliner import GLiNER
model = GLiNER.from_pretrained("knowledgator/gliner-x-large")
text = """
Cristiano Ronaldo dos Santos Aveiro (pronunciación en portugués: [kɾiʃˈtjɐnu ʁɔˈnaldu]; nacido el 5 de febrero de 1985) es un futbolista profesional portugués que juega como delantero y es capitán tanto del club Al Nassr de la Liga Profesional Saudí como de la selección nacional de Portugal. Ampliamente considerado como uno de los mejores jugadores de todos los tiempos, Ronaldo ha ganado cinco premios Balón de Oro, un récord de tres premios al Jugador del Año de la UEFA y cuatro Botas de Oro europeas, la mayor cantidad para un jugador europeo. Ha ganado 33 trofeos en su carrera, incluidos siete títulos de liga, cinco Ligas de Campeones de la UEFA, la Eurocopa y la Liga de Naciones de la UEFA. Ronaldo posee los récords de más apariciones (183), goles (140) y asistencias (42) en la Liga de Campeones, goles en la Eurocopa (14), goles internacionales (128) y apariciones internacionales (205). Es uno de los pocos jugadores que ha disputado más de 1.200 partidos profesionales en su carrera, la mayor cantidad para un jugador de campo, y ha marcado más de 850 goles oficiales en su carrera senior con clubes y selecciones, lo que lo convierte en el máximo goleador de todos los tiempos.
"""
labels = ["persona", "premio", "fecha", "competiciones", "equipos"]
entities = model.predict_entities(text, labels, threshold=0.5)
for entity in entities:
    print(entity["text"], "=>", entity["label"])
```

```
Cristiano Ronaldo dos Santos Aveiro => persona
5 de febrero de 1985 => fecha
Al Nassr de la Liga Profesional Saudí => equipos
selección nacional de Portugal => equipos
Balón de Oro => premio
Jugador del Año de la UEFA => premio
Botas de Oro europeas => premio
33 trofeos => premio
Ligas de Campeones de la UEFA => competiciones
Eurocopa => competiciones
Liga de Naciones de la UEFA => competiciones
Liga de Campeones => competiciones
```
</details>

<details>
<summary>Danish</summary>

```python
from gliner import GLiNER
model = GLiNER.from_pretrained("knowledgator/gliner-x-large")
text = """
Cristiano Ronaldo dos Santos Aveiro (portugisisk udtale: [kɾiʃˈtjɐnu ʁɔˈnaldu]; født 5. februar 1985) er en portugisisk professionel fodboldspiller, der spiller som angriber for og er anfører for både den saudiske Pro League-klub Al Nassr og det portugisiske landshold. Bredt anerkendt som en af de største spillere gennem tiderne har Ronaldo vundet fem Ballon d'Or-priser, en rekord på tre UEFA Men's Player of the Year-priser og fire europæiske Guldstøvler – flest af en europæisk spiller. Han har vundet 33 trofæer i sin karriere, herunder syv ligatitler, fem UEFA Champions League-titler, EM og UEFA Nations League. Ronaldo har rekorderne for flest optrædener (183), mål (140) og assists (42) i Champions League, mål ved EM (14), internationale mål (128) og internationale optrædener (205). Han er en af de få spillere, der har spillet over 1.200 professionelle kampe – flest af en markspiller – og har scoret over 850 officielle mål i sin seniorkarriere for klub og land, hvilket gør ham til historiens mest scorende spiller.
"""
labels = ["person", "pris", "dato", "turneringer", "hold"]
entities = model.predict_entities(text, labels, threshold=0.5)
for entity in entities:
    print(entity["text"], "=>", entity["label"])
```

```
Cristiano Ronaldo dos Santos Aveiro => person
5. februar 1985 => dato
Pro League-klub => hold
Al Nassr => hold
portugisiske landshold => hold
Ballon d'Or-priser => pris
UEFA Men's Player of the Year-priser => turneringer
Guldstøvler => pris
UEFA Champions League-titler => turneringer
EM => turneringer
UEFA Nations League => turneringer
Champions League => turneringer
EM => turneringer
```
</details>

<details>
<summary>Swedish</summary>

```python
from gliner import GLiNER
model = GLiNER.from_pretrained("knowledgator/gliner-x-large")
text = """
Cristiano Ronaldo dos Santos Aveiro (portugisisk uttal: [kɾiʃˈtjɐnu ʁɔˈnaldu]; född 5 februari 1985) är en portugisisk professionell fotbollsspelare som spelar som anfallare för och är kapten för både Saudi Pro League-klubben Al Nassr och Portugals landslag. Allmänt ansedd som en av de största spelarna genom tiderna har Ronaldo vunnit fem Ballon d'Or-utmärkelser, ett rekord på tre UEFA:s Årets spelare och fyra europeiska Gyllene skor – flest av en europeisk spelare. Han har vunnit 33 troféer under sin karriär, inklusive sju ligatitlar, fem UEFA Champions League-titlar, UEFA:s europamästerskap och UEFA Nations League. Ronaldo innehar rekorden för flest framträdanden (183), mål (140) och assist (42) i Champions League, mål i EM (14), landslagsmål (128) och landslagsframträdanden (205). Han är en av få spelare som spelat över 1 200 professionella matcher, flest av en utespelare, och har gjort över 850 officiella seniormål för klubb och landslag, vilket gör honom till den främsta målskytten genom tiderna.
"""
labels = ["person", "utmärkelse", "datum", "tävlingar", "lag"]
entities = model.predict_entities(text, labels, threshold=0.5)
for entity in entities:
    print(entity["text"], "=>", entity["label"])
```

```
Cristiano Ronaldo dos Santos Aveiro => person
5 februari 1985 => datum
Saudi Pro League-klubben => lag
Al Nassr => lag
Portugals landslag => lag
Ballon d'Or-utmärkelser => utmärkelse
Årets spelare => utmärkelse
Gyllene skor => utmärkelse
33 troféer => utmärkelse
sju ligatitlar => utmärkelse
UEFA Champions League-titlar => tävlingar
UEFA:s europamästerskap => tävlingar
UEFA Nations League => tävlingar
Champions League => tävlingar
EM => tävlingar
```
</details>

<details>
<summary>Norwegian</summary>

```python
from gliner import GLiNER
model = GLiNER.from_pretrained("knowledgator/gliner-x-large")
text = """
Cristiano Ronaldo dos Santos Aveiro (portugisisk uttale: [kɾiʃˈtjɐnu ʁɔˈnaldu]; født 5. februar 1985) er en portugisisk profesjonell fotballspiller som spiller som spiss og er kaptein både for den saudiarabiske klubben Al Nassr og det portugisiske landslaget. Bredt ansett som en av tidenes beste spillere, har Ronaldo vunnet fem Ballon d'Or-priser, en rekord på tre UEFA Årets Spiller-priser, og fire europeiske Gullstøvler – flest av alle europeiske spillere. Han har vunnet 33 troféer i løpet av karrieren, inkludert syv ligatitler, fem UEFA Champions League-titler, EM og UEFA Nations League. Ronaldo innehar rekordene for flest opptredener (183), mål (140) og målgivende pasninger (42) i Champions League, mål i EM (14), landslagsmål (128) og landskamper (205). Han er en av få spillere med over 1200 profesjonelle kamper, flest for en utespiller, og har scoret over 850 mål i offisielle seniorkamper for klubb og landslag, noe som gjør ham til tidenes toppscorer.
"""
labels = ["person", "pris", "dato", "konkurranser", "lag"]
entities = model.predict_entities(text, labels, threshold=0.5)
for entity in entities:
    print(entity["text"], "=>", entity["label"])
```

```
Cristiano Ronaldo dos Santos Aveiro => person
5. februar 1985 => dato
Al Nassr => lag
portugisiske landslaget => lag
Ballon d'Or-priser => pris
UEFA Årets Spiller-priser => konkurranser
Gullstøvler => konkurranser
33 troféer => pris
syv ligatitler => pris
UEFA Champions League-titler => konkurranser
EM => konkurranser
UEFA Nations League => konkurranser
Champions League => konkurranser
EM => konkurranser
```
</details>

<details>
<summary>Czech</summary>

```python
from gliner import GLiNER
model = GLiNER.from_pretrained("knowledgator/gliner-x-large")
text = """
Cristiano Ronaldo dos Santos Aveiro (portugalská výslovnost: [kɾiʃˈtjɐnu ʁɔˈnaldu]; narozen 5. února 1985) je portugalský profesionální fotbalista, který hraje jako útočník a je kapitánem jak klubu Al Nassr v Saúdské profesionální lize, tak portugalského národního týmu. Široce považován za jednoho z nejlepších hráčů všech dob, Ronaldo získal pět ocenění Ballon d'Or, rekordní tři ocenění UEFA Hráč roku a čtyři Zlaté kopačky, což je nejvíce ze všech evropských hráčů. Ve své kariéře vyhrál 33 trofejí, včetně sedmi ligových titulů, pěti Lig mistrů UEFA, Mistrovství Evropy UEFA a Ligy národů UEFA. Ronaldo drží rekordy v počtu startů (183), gólů (140) a asistencí (42) v Lize mistrů, gólů na mistrovství Evropy (14), mezinárodních gólů (128) a mezinárodních startů (205). Je jedním z mála hráčů, kteří odehráli více než 1 200 profesionálních zápasů, což je nejvíce mezi hráči v poli, a vstřelil přes 850 oficiálních gólů na klubové a reprezentační úrovni, čímž se stal nejlepším střelcem všech dob.
"""
labels = ["osoba", "ocenění", "datum", "soutěže", "týmy"]
entities = model.predict_entities(text, labels, threshold=0.5)
for entity in entities:
    print(entity["text"], "=>", entity["label"])
```

```
Cristiano Ronaldo dos Santos Aveiro => osoba
5. února 1985 => datum
Al Nassr => týmy
Saúdské profesionální lize => týmy
Ballon d'Or => ocenění
rekordní tři ocenění UEFA Hráč roku => ocenění
Zlaté kopačky => ocenění
33 trofejí => ocenění
sedmi ligových titulů => ocenění
Lig mistrů UEFA => soutěže
Mistrovství Evropy UEFA => soutěže
Ligy národů UEFA => soutěže
Lize mistrů => soutěže
mistrovství Evropy => soutěže
```
</details>

<details>
<summary>Polish</summary>

```python
from gliner import GLiNER
model = GLiNER.from_pretrained("knowledgator/gliner-x-large")
text = """
Cristiano Ronaldo dos Santos Aveiro (wymowa portugalska: [kɾiʃˈtjɐnu ʁɔˈnaldu]; ur. 5 lutego 1985) to portugalski piłkarz grający na pozycji napastnika, kapitan klubu Al Nassr z saudyjskiej ligi oraz reprezentacji Portugalii. Uważany za jednego z najwybitniejszych zawodników w historii, Ronaldo zdobył pięć Złotych Piłek, rekordowe trzy nagrody UEFA dla najlepszego piłkarza roku oraz cztery Złote Buty, najwięcej wśród europejskich zawodników. W swojej karierze zdobył 33 trofea, w tym siedem tytułów mistrza ligi, pięć Lig Mistrzów UEFA, mistrzostwo Europy i Ligę Narodów UEFA. Ronaldo posiada rekordy w liczbie występów (183), goli (140) i asyst (42) w Lidze Mistrzów, bramek na Mistrzostwach Europy (14), goli międzynarodowych (128) oraz meczów międzynarodowych (205). Jest jednym z nielicznych piłkarzy z ponad 1200 oficjalnymi występami w karierze — najwięcej spośród graczy z pola — oraz zdobywcą ponad 850 goli dla klubów i reprezentacji, co czyni go najlepszym strzelcem wszech czasów.
"""
labels = ["osoba", "nagroda", "data", "rozgrywki", "drużyny"]
entities = model.predict_entities(text, labels, threshold=0.5)
for entity in entities:
    print(entity["text"], "=>", entity["label"])
```

```
Cristiano Ronaldo dos Santos Aveiro => osoba
5 lutego 1985 => data
Al Nassr => drużyny
reprezentacji Portugalii => drużyny
Złotych Piłek => nagroda
Złote Buty => nagroda
Lig Mistrzów UEFA => rozgrywki
mistrzostwo Europy => rozgrywki
Ligę Narodów UEFA => rozgrywki
Mistrzostwach Europy => rozgrywki
```
</details>

<details>
<summary>Lithuanian</summary>

```python
from gliner import GLiNER
model = GLiNER.from_pretrained("knowledgator/gliner-x-large")
text = """
Cristiano Ronaldo dos Santos Aveiro (portugališka tarimo forma: [kɾiʃˈtjɐnu ʁɔˈnaldu]; gimė 1985 m. vasario 5 d.) yra portugalų profesionalus futbolininkas, žaidžiantis puolėjo pozicijoje ir esantis tiek Saudo Arabijos „Pro League“ klubo „Al Nassr“, tiek Portugalijos nacionalinės rinktinės kapitonas. Plačiai laikomas vienu geriausių visų laikų žaidėjų, Ronaldo yra laimėjęs penkis „Ballon d'Or“ apdovanojimus, rekordiškai tris UEFA metų žaidėjo apdovanojimus ir keturis Europos „Auksinius batelius“ – daugiausiai tarp Europos žaidėjų. Savo karjeroje jis laimėjo 33 trofėjus, įskaitant septynis lygos titulus, penkis UEFA Čempionų lygos titulus, UEFA Europos čempionatą ir UEFA Tautų lygą. Ronaldo priklauso rekordai pagal daugiausiai pasirodymų (183), įvarčių (140) ir rezultatyvių perdavimų (42) Čempionų lygoje, įvarčių Europos čempionate (14), tarptautinių įvarčių (128) ir tarptautinių pasirodymų (205). Jis yra vienas iš nedaugelio žaidėjų, sužaidusių daugiau nei 1200 profesionalių rungtynių – daugiausiai tarp aikštės žaidėjų – ir pelnęs daugiau nei 850 oficialių įvarčių klubų ir rinktinės lygiu, tapdamas rezultatyviausiu visų laikų žaidėju.
"""
labels = ["asmuo", "apdovanojimas", "data", "varžybos", "komandos"]
entities = model.predict_entities(text, labels, threshold=0.5)
for entity in entities:
    print(entity["text"], "=>", entity["label"])
```

```
Cristiano Ronaldo dos Santos Aveiro => asmuo
1985 m. vasario 5 d. => data
Al Nassr => komandos
Ballon d'Or => apdovanojimas
UEFA metų žaidėjo apdovanojimus => apdovanojimas
Europos „Auksinius batelius => apdovanojimas
UEFA Čempionų lygos => varžybos
UEFA Europos čempionatą => varžybos
UEFA Tautų lygą => varžybos
```
</details>

<details>
<summary>Estonian</summary>

```python
from gliner import GLiNER
model = GLiNER.from_pretrained("knowledgator/gliner-x-large")
text = """
Cristiano Ronaldo dos Santos Aveiro (portugali hääldus: [kɾiʃˈtjɐnu ʁɔˈnaldu]; sündinud 5. veebruaril 1985) on Portugali elukutseline jalgpallur, kes mängib ründajana ja on kapteniks nii Saudi Araabia Pro League'i klubis Al Nassr kui ka Portugali rahvuskoondises. Teda peetakse laialdaselt üheks aegade parimaks mängijaks. Ronaldo on võitnud viis Ballon d'Or auhinda, kolm UEFA Aasta Meesmängija auhinda ning neli Euroopa Kuldset Saapa auhinda – enim Euroopa mängijate seas. Ta on oma karjääri jooksul võitnud 33 trofeed, sealhulgas seitse liigatiitlit, viis UEFA Meistrite Liigat, UEFA Euroopa meistrivõistlused ja UEFA Rahvuste Liiga. Ronaldol on Meistrite Liigas enim mänge (183), väravaid (140) ja resultatiivseid sööte (42), Euroopa meistrivõistlustel enim väravaid (14), rahvusvahelisi väravaid (128) ja rahvusvahelisi mänge (205). Ta on üks vähestest mängijatest, kes on pidanud üle 1200 ametliku mängu ning löönud üle 850 värava klubide ja koondise eest, olles kõigi aegade parim väravakütt.
"""
labels = ["isik", "auhind", "kuupäev", "võistlused", "meeskonnad"]
entities = model.predict_entities(text, labels, threshold=0.5)
for entity in entities:
    print(entity["text"], "=>", entity["label"])
```

```
Cristiano Ronaldo dos Santos Aveiro => isik
5. veebruaril 1985 => kuupäev
Al Nassr => meeskonnad
Portugali rahvuskoondises => meeskonnad
Ballon d'Or auhinda => auhind
UEFA Aasta Meesmängija auhinda => auhind
Euroopa Kuldset Saapa auhinda => auhind
UEFA Meistrite Liigat => võistlused
UEFA Euroopa meistrivõistlused => võistlused
UEFA Rahvuste Liiga => võistlused
```
</details>

<details>
<summary>Latvian</summary>

```python
from gliner import GLiNER
model = GLiNER.from_pretrained("knowledgator/gliner-x-large")
text = """
Kristiānu Ronaldu dušu Santušu Aveiru (portugāļu izruna: [kɾiʃˈtjɐnu ʁɔˈnaldu]; dzimis 1985. gada 5. februārī) ir portugāļu profesionāls futbolists, kurš spēlē kā uzbrucējs un ir kapteinis gan Saūda Arābijas Pro līgas klubā "Al Nassr", gan Portugāles izlasē. Plaši tiek uzskatīts par vienu no visu laiku izcilākajiem spēlētājiem. Ronaldu ir ieguvis piecas Ballon d'Or balvas, rekorda trīs UEFA Gada spēlētāja balvas un četras Eiropas Zelta bučus – visvairāk starp Eiropas spēlētājiem. Viņš savas karjeras laikā ir izcīnījis 33 trofejas, tostarp septiņus līgas titulus, piecas UEFA Čempionu līgas, UEFA Eiropas čempionātu un UEFA Nāciju līgu. Ronaldu pieder rekordi pēc dalību skaita (183), vārtu guvumiem (140) un piespēlēm (42) Čempionu līgā, vārtiem Eiropas čempionātā (14), vārtiem starptautiskā līmenī (128) un spēļu skaita izlasē (205). Viņš ir viens no nedaudzajiem spēlētājiem ar vairāk nekā 1200 profesionālām spēlēm, visvairāk starp laukuma spēlētājiem, un ir guvis vairāk nekā 850 oficiālus vārtus klubos un izlasē, padarot viņu par visu laiku rezultatīvāko spēlētāju.
"""
labels = ["persona", "balva", "datums", "sacensības", "komandas"]
entities = model.predict_entities(text, labels, threshold=0.5)
for entity in entities:
    print(entity["text"], "=>", entity["label"])
```

```
Kristiānu Ronaldu => persona
Santušu Aveiru => persona
1985. gada 5. februārī => datums
Al Nassr => komandas
Portugāles izlasē => komandas
Ballon d'Or balvas => balva
UEFA Gada spēlētāja balvas => balva
Eiropas Zelta bučus => balva
UEFA Čempionu līgas => sacensības
UEFA Eiropas čempionātu => sacensības
UEFA Nāciju līgu => sacensības
```
</details>

<details>
<summary>Finnish</summary>

```python
from gliner import GLiNER
model = GLiNER.from_pretrained("knowledgator/gliner-x-large")
text = """
Cristiano Ronaldo dos Santos Aveiro (lausutaan portugaliksi: [kɾiʃˈtjɐnu ʁɔˈnaldu]; syntynyt 5. helmikuuta 1985) on portugalilainen ammattilaisjalkapalloilija, joka pelaa hyökkääjänä ja toimii kapteenina sekä Saudi Pro League -seura Al Nassrissa että Portugalin maajoukkueessa. Häntä pidetään laajalti yhtenä kaikkien aikojen parhaista pelaajista. Ronaldo on voittanut viisi Ballon d'Or -palkintoa, ennätykselliset kolme UEFA:n Vuoden Pelaaja -palkintoa ja neljä Euroopan Kultakenkää – eniten eurooppalaispelaajista. Hän on urallaan voittanut 33 pokaalia, mukaan lukien seitsemän sarjamestaruutta, viisi UEFA Mestarien liigaa, UEFA:n Euroopan-mestaruuden ja UEFA Nations Leaguen. Ronaldo pitää hallussaan ennätyksiä Mestarien liigassa pelien (183), maalien (140) ja syöttöjen (42) määrässä, EM-kisojen maaleissa (14), maaottelumaaleissa (128) ja maaotteluiden määrässä (205). Hän on yksi harvoista pelaajista, joka on pelannut yli 1 200 ammattilaisottelua – eniten kenttäpelaajista – ja tehnyt yli 850 virallista maalia seurassa ja maajoukkueessa, mikä tekee hänestä kaikkien aikojen maalikuninkaan.
"""
labels = ["henkilö", "palkinto", "päivämäärä", "kilpailut", "joukkueet"]
entities = model.predict_entities(text, labels, threshold=0.5)
for entity in entities:
    print(entity["text"], "=>", entity["label"])
```

```
Cristiano Ronaldo dos Santos Aveiro => henkilö
5. helmikuuta 1985 => päivämäärä
Saudi Pro League => joukkueet
Al Nassrissa => joukkueet
Portugalin maajoukkueessa => joukkueet
Ballon d'Or -palkintoa => palkinto
UEFA:n Vuoden Pelaaja -palkintoa => palkinto
Euroopan Kultakenkää => palkinto
UEFA Mestarien liigaa => kilpailut
UEFA:n Euroopan-mestaruuden => kilpailut
UEFA Nations Leaguen => kilpailut
EM-kisojen => kilpailut
```
</details>

<details>
<summary>German</summary>

```python
from gliner import GLiNER
model = GLiNER.from_pretrained("knowledgator/gliner-x-large")
text = """
Cristiano Ronaldo dos Santos Aveiro (portugiesische Aussprache: [kɾiʃˈtjɐnu ʁɔˈnaldu]; geboren am 5. Februar 1985) ist ein portugiesischer Profifußballer, der als Stürmer spielt und sowohl für den Verein Al Nassr in der Saudi Pro League als auch für die portugiesische Nationalmannschaft Kapitän ist. Allgemein gilt er als einer der größten Spieler aller Zeiten. Ronaldo hat fünf Ballon-d'Or-Auszeichnungen, einen Rekord von drei UEFA-Auszeichnungen als Spieler des Jahres und vier europäische Goldene Schuhe gewonnen – die meisten für einen europäischen Spieler. In seiner Karriere hat er 33 Trophäen gewonnen, darunter sieben Ligatitel, fünf UEFA-Champions-League-Titel, die UEFA-Europameisterschaft und die UEFA Nations League. Ronaldo hält die Rekorde für die meisten Einsätze (183), Tore (140) und Assists (42) in der Champions League, Tore bei Europameisterschaften (14), Länderspieltore (128) und Länderspieleinsätze (205). Er ist einer der wenigen Spieler, die über 1.200 Einsätze in ihrer Profikarriere absolviert haben – die meisten eines Feldspielers – und hat über 850 offizielle Tore für Verein und Land erzielt, womit er der erfolgreichste Torschütze aller Zeiten ist.
"""
labels = ["person", "auszeichnung", "datum", "wettbewerbe", "teams"]
entities = model.predict_entities(text, labels, threshold=0.5)
for entity in entities:
    print(entity["text"], "=>", entity["label"])
```

```
Cristiano Ronaldo dos Santos Aveiro => person
5. Februar 1985 => datum
Al Nassr => teams
Saudi Pro League => wettbewerbe
portugiesische Nationalmannschaft => teams
Ballon-d'Or-Auszeichnungen => auszeichnung
UEFA-Auszeichnungen => auszeichnung
Spieler des Jahres => auszeichnung
europäische Goldene Schuhe => auszeichnung
UEFA-Champions-League-Titel => wettbewerbe
UEFA-Europameisterschaft => wettbewerbe
UEFA Nations League => wettbewerbe
```
</details>

<details>
<summary>French</summary>

```python
from gliner import GLiNER
model = GLiNER.from_pretrained("knowledgator/gliner-x-large")
text = """
Cristiano Ronaldo dos Santos Aveiro (prononciation portugaise : [kɾiʃˈtjɐnu ʁɔˈnaldu] ; né le 5 février 1985) est un footballeur professionnel portugais qui joue comme attaquant et est capitaine à la fois du club d'Al Nassr en Saudi Pro League et de l'équipe nationale du Portugal. Largement considéré comme l’un des plus grands joueurs de tous les temps, Ronaldo a remporté cinq Ballons d’Or, un record de trois prix du Joueur de l’année UEFA et quatre Souliers d’or européens, le plus grand nombre pour un joueur européen. Il a remporté 33 trophées dans sa carrière, dont sept titres de championnat, cinq Ligues des champions de l’UEFA, le Championnat d'Europe et la Ligue des nations de l’UEFA. Ronaldo détient les records du plus grand nombre d'apparitions (183), de buts (140) et de passes décisives (42) en Ligue des champions, de buts dans le Championnat d'Europe (14), de buts internationaux (128) et d'apparitions internationales (205). Il est l’un des rares joueurs à avoir disputé plus de 1 200 matchs professionnels en carrière, le plus grand nombre pour un joueur de champ, et a marqué plus de 850 buts officiels en carrière en club et en sélection, ce qui fait de lui le meilleur buteur de tous les temps.
"""
labels = ["personne", "récompense", "date", "compétitions", "équipes"]
entities = model.predict_entities(text, labels, threshold=0.5)
for entity in entities:
    print(entity["text"], "=>", entity["label"])
```

```
Cristiano Ronaldo dos Santos Aveiro => personne
5 février 1985 => date
Al Nassr => équipes
Saudi Pro League => compétitions
équipe nationale du Portugal => équipes
Ballons d’Or => récompense
Joueur de l’année UEFA => récompense
Souliers d’or européens => récompense
Ligues des champions de l’UEFA => compétitions
Championnat d'Europe => compétitions
Ligue des nations de l’UEFA => compétitions
```
</details>

<details>
<summary>Italian</summary>

```python
from gliner import GLiNER
model = GLiNER.from_pretrained("knowledgator/gliner-x-large")
text = """
Cristiano Ronaldo dos Santos Aveiro (pronuncia portoghese: [kɾiʃˈtjɐnu ʁɔˈnaldu]; nato il 5 febbraio 1985) è un calciatore professionista portoghese che gioca come attaccante e che è capitano sia del club Al Nassr della Saudi Pro League sia della nazionale portoghese. Considerato ampiamente uno dei più grandi giocatori di tutti i tempi, Ronaldo ha vinto cinque Palloni d'Oro, un record di tre premi Giocatore dell'Anno UEFA e quattro Scarpe d'Oro europee, il numero più alto per un giocatore europeo. Ha vinto 33 trofei nella sua carriera, inclusi sette titoli di campionato, cinque Champions League UEFA, il Campionato europeo UEFA e la UEFA Nations League. Ronaldo detiene i record per il maggior numero di presenze (183), gol (140) e assist (42) in Champions League, gol nel Campionato europeo (14), gol internazionali (128) e presenze internazionali (205). È uno dei pochi giocatori ad aver superato le 1.200 presenze professionistiche in carriera, il numero più alto per un giocatore di movimento, e ha segnato oltre 850 gol ufficiali con club e nazionale, rendendolo il miglior marcatore di tutti i tempi.
"""
labels = ["persona", "premio", "data", "competizioni", "squadre"]
entities = model.predict_entities(text, labels, threshold=0.5)
for entity in entities:
    print(entity["text"], "=>", entity["label"])
```

```
Cristiano Ronaldo dos Santos Aveiro => persona
5 febbraio 1985 => data
Al Nassr => squadre
Saudi Pro League => competizioni
Palloni d'Oro => premio
Giocatore dell'Anno UEFA => premio
Scarpe d'Oro europee => premio
Champions League UEFA => competizioni
Campionato europeo UEFA => competizioni
UEFA Nations League => competizioni
```
</details>

<details>
<summary>Romanian</summary>

```python
from gliner import GLiNER
model = GLiNER.from_pretrained("knowledgator/gliner-x-large")
text = """
Cristiano Ronaldo dos Santos Aveiro (pronunție în portugheză: [kɾiʃˈtjɐnu ʁɔˈnaldu]; născut la 5 februarie 1985) este un fotbalist profesionist portughez care joacă pe postul de atacant și este căpitan atât al clubului Al Nassr din Liga Profesionistă Saudită, cât și al echipei naționale a Portugaliei. Larg considerat unul dintre cei mai buni jucători din toate timpurile, Ronaldo a câștigat cinci premii Ballon d'Or, un record de trei premii Jucătorul Anului UEFA și patru Ghete de Aur europene, cele mai multe pentru un jucător european. A câștigat 33 de trofee în carieră, inclusiv șapte titluri de campionat, cinci Ligi ale Campionilor UEFA, Campionatul European UEFA și Liga Națiunilor UEFA. Ronaldo deține recordurile pentru cele mai multe apariții (183), goluri (140) și pase decisive (42) în Liga Campionilor, goluri în Campionatul European (14), goluri internaționale (128) și apariții internaționale (205). Este unul dintre puținii jucători care au jucat peste 1.200 de meciuri profesioniste, cele mai multe pentru un jucător de câmp, și a marcat peste 850 de goluri oficiale în carieră pentru club și țară, devenind cel mai bun marcator din toate timpurile.
"""
labels = ["persoană", "premiu", "dată", "competiții", "echipe"]
entities = model.predict_entities(text, labels, threshold=0.5)
for entity in entities:
    print(entity["text"], "=>", entity["label"])
```

```
Cristiano Ronaldo dos Santos Aveiro => persoană
5 februarie 1985 => dată
Al Nassr => echipe
Liga Profesionistă Saudită => competiții
Ballon d'Or => premiu
Jucătorul Anului UEFA => premiu
Ghete de Aur europene => premiu
Ligi ale Campionilor UEFA => competiții
Campionatul European UEFA => competiții
Liga Națiunilor UEFA => competiții
```
</details>

<details>
<summary>Portuguese</summary>

```python
from gliner import GLiNER
model = GLiNER.from_pretrained("knowledgator/gliner-x-large")
text = """
Cristiano Ronaldo dos Santos Aveiro (pronúncia em português: [kɾiʃˈtjɐnu ʁɔˈnaldu]; nascido em 5 de fevereiro de 1985) é um futebolista profissional português que atua como atacante e é capitão tanto do clube Al Nassr da Liga Profissional Saudita quanto da seleção nacional de Portugal. Amplamente considerado um dos maiores jogadores de todos os tempos, Ronaldo conquistou cinco prêmios Ballon d'Or, um recorde de três prêmios de Jogador do Ano da UEFA e quatro Chuteiras de Ouro europeias, o maior número para um jogador europeu. Ele venceu 33 troféus em sua carreira, incluindo sete títulos de liga, cinco Ligas dos Campeões da UEFA, o Campeonato Europeu da UEFA e a Liga das Nações da UEFA. Ronaldo detém os recordes de mais aparições (183), gols (140) e assistências (42) na Liga dos Campeões, gols no Campeonato Europeu (14), gols internacionais (128) e aparições internacionais (205). Ele é um dos poucos jogadores a ter feito mais de 1.200 aparições profissionais, o maior número entre jogadores de linha, e marcou mais de 850 gols oficiais na carreira profissional por clubes e pela seleção, sendo o maior artilheiro de todos os tempos.
"""
labels = ["pessoa", "prêmio", "data", "competições", "equipes"]
entities = model.predict_entities(text, labels, threshold=0.5)
for entity in entities:
    print(entity["text"], "=>", entity["label"])
```

```
Cristiano Ronaldo dos Santos Aveiro => pessoa
5 de fevereiro de 1985 => data
Al Nassr => equipes
Liga Profissional Saudita => competições
seleção nacional de Portugal => equipes
Ballon d'Or => prêmio
Jogador do Ano da UEFA => prêmio
Chuteiras de Ouro europeias => prêmio
Ligas dos Campeões da UEFA => competições
Campeonato Europeu da UEFA => competições
Liga das Nações da UEFA => competições
```
</details>

<details>
<summary>Portuguese</summary>

```python
from gliner import GLiNER
model = GLiNER.from_pretrained("knowledgator/gliner-x-large")
text = """
Cristiano Ronaldo dos Santos Aveiro (Portugese uitspraak: [kɾiʃˈtjɐnu ʁɔˈnaldu]; geboren op 5 februari 1985) is een Portugese professionele voetballer die als aanvaller speelt voor en aanvoerder is van zowel de Saoedische Pro League-club Al Nassr als het nationale team van Portugal. Algemeen beschouwd als een van de beste spelers aller tijden, heeft Ronaldo vijf Ballon d'Or-onderscheidingen gewonnen, een record van drie UEFA Men's Player of the Year Awards, en vier Europese Gouden Schoenen, het meeste voor een Europese speler. Hij heeft 33 trofeeën gewonnen in zijn carrière, waaronder zeven landstitels, vijf UEFA Champions Leagues, het EK en de UEFA Nations League. Ronaldo bezit de records voor de meeste optredens (183), doelpunten (140) en assists (42) in de Champions League, doelpunten op het EK (14), interlanddoelpunten (128) en interlandoptredens (205). Hij is een van de weinige spelers die meer dan 1.200 professionele wedstrijden heeft gespeeld, het meeste voor een veldspeler, en heeft meer dan 850 officiële doelpunten gescoord voor club en land, waarmee hij de topscorer aller tijden is.
"""
labels = ["persoon", "onderscheiding", "datum", "competities", "teams"]
entities = model.predict_entities(text, labels, threshold=0.5)
for entity in entities:
    print(entity["text"], "=>", entity["label"])
```

```
Cristiano Ronaldo dos Santos Aveiro => persoon
5 februari 1985 => datum
Saoedische Pro League-club => teams
Al Nassr => teams
nationale team van Portugal => teams
Ballon d'Or-onderscheidingen => onderscheiding
UEFA Men's Player of the Year Awards => onderscheiding
Europese Gouden Schoenen => onderscheiding
UEFA Champions Leagues => competities
EK => competities
UEFA Nations League => competities
```
</details>
