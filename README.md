# Wine recommender system
Research questions 

Does wine descriptors impact Rating ?
  - Machine learning to classify reviews 
     what are key words that determines ratings?
     
    Apply Logistic regression, Random Forest to determine relationship between key words and ratings, wine type, varieties  
        
 There is a correlation between Ratings and Price ?
   - Correlation analysis


The higher the Rating and Price the higher the consumption of the wine variety
   - Linear Regression

 Are wine descriptors good predictors of  wine price and points rating ?

#----------------------------------------------------------------------------
```python
import numpy as np

import csv
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats

pd.options.display.max_rows = 999
pd.options.display.max_columns = 999

from sklearn import cluster
from sklearn import metrics
from sklearn.metrics import pairwise_distances


import seaborn as sns




```python
# path='C:/Users/bless/Documents/GitHub/projects/project-capstone/Capstone project data'
# alternate file winemag-data-130k-v2.csv

wine = pd.read_csv('C:/Users/bless/Documents/GitHub/projects/project-capstone/Wine/wine-reviews/winemag-data_first150k - Copy.csv',\
                   index_col=0,parse_dates=True,   dayfirst=True,encoding='utf-8')

```


```python
wine.head(5)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>description</th>
      <th>designation</th>
      <th>points</th>
      <th>price</th>
      <th>province</th>
      <th>region_1</th>
      <th>region_2</th>
      <th>variety</th>
      <th>winery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>US</td>
      <td>This tremendous 100% varietal wine hails from ...</td>
      <td>Martha's Vineyard</td>
      <td>96</td>
      <td>235.0</td>
      <td>California</td>
      <td>Napa Valley</td>
      <td>Napa</td>
      <td>Cabernet Sauvignon</td>
      <td>Heitz</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Spain</td>
      <td>Ripe aromas of fig, blackberry and cassis are ...</td>
      <td>Carodorum Selección Especial Reserva</td>
      <td>96</td>
      <td>110.0</td>
      <td>Northern Spain</td>
      <td>Toro</td>
      <td>NaN</td>
      <td>Tinta de Toro</td>
      <td>Bodega Carmen Rodríguez</td>
    </tr>
    <tr>
      <th>2</th>
      <td>US</td>
      <td>Mac Watson honors the memory of a wine once ma...</td>
      <td>Special Selected Late Harvest</td>
      <td>96</td>
      <td>90.0</td>
      <td>California</td>
      <td>Knights Valley</td>
      <td>Sonoma</td>
      <td>Sauvignon Blanc</td>
      <td>Macauley</td>
    </tr>
    <tr>
      <th>3</th>
      <td>US</td>
      <td>This spent 20 months in 30% new French oak, an...</td>
      <td>Reserve</td>
      <td>96</td>
      <td>65.0</td>
      <td>Oregon</td>
      <td>Willamette Valley</td>
      <td>Willamette Valley</td>
      <td>Pinot Noir</td>
      <td>Ponzi</td>
    </tr>
    <tr>
      <th>4</th>
      <td>France</td>
      <td>This is the top wine from La Bégude, named aft...</td>
      <td>La Brûlade</td>
      <td>95</td>
      <td>66.0</td>
      <td>Provence</td>
      <td>Bandol</td>
      <td>NaN</td>
      <td>Provence red blend</td>
      <td>Domaine de la Bégude</td>
    </tr>
  </tbody>
</table>
</div>


```python
wine.variety.unique()
```
    array(['Cabernet Sauvignon', 'Tinta de Toro', 'Sauvignon Blanc',
           'Pinot Noir', 'Provence red blend', 'Friulano', 'Tannat',
           'Chardonnay', 'Tempranillo', 'Malbec', 'Rosé', 'Tempranillo Blend',
           'Syrah', 'Mavrud', 'Sangiovese', 'Sparkling Blend',
           'Rhône-style White Blend', 'Red Blend', 'Mencía', 'Palomino',
           'Petite Sirah', 'Riesling', 'Cabernet Sauvignon-Syrah',
           'Portuguese Red', 'Nebbiolo', 'Pinot Gris', 'Meritage', 'Baga',
           'Glera', 'Malbec-Merlot', 'Merlot-Malbec', 'Ugni Blanc-Colombard',
           'Viognier', 'Cabernet Sauvignon-Cabernet Franc', 'Moscato',
           'Pinot Grigio', 'Cabernet Franc', 'White Blend', 'Monastrell',
           'Gamay', 'Zinfandel', 'Greco', 'Barbera', 'Grenache',
           'Rhône-style Red Blend', 'Albariño', 'Malvasia Bianca',
           'Assyrtiko', 'Malagouzia', 'Carmenère', 'Bordeaux-style Red Blend',
           'Touriga Nacional', 'Agiorgitiko', 'Picpoul', 'Godello',
           'Gewürztraminer', 'Merlot', 'Syrah-Grenache', 'G-S-M', 'Mourvèdre',
           'Bordeaux-style White Blend', 'Petit Verdot', 'Muscat',
           'Chenin Blanc-Chardonnay', 'Cabernet Sauvignon-Merlot',
           'Pinot Bianco', 'Alvarinho', 'Portuguese White', 'Garganega',
           'Sauvignon', 'Gros and Petit Manseng', 'Tannat-Cabernet',
           'Alicante Bouschet', 'Aragonês', 'Silvaner', 'Ugni Blanc',
           'Grüner Veltliner', 'Frappato', 'Lemberger', 'Sylvaner',
           'Chasselas', 'Alsace white blend', 'Früburgunder', 'Kekfrankos',
           'Vermentino', 'Sherry', 'Aglianico', 'Torrontés', 'Primitivo',
           'Semillon-Sauvignon Blanc', 'Portuguese Rosé', 'Grenache-Syrah',
           'Prié Blanc', 'Negrette', 'Furmint', 'Carignane', 'Pinot Blanc',
           "Nero d'Avola", 'St. Laurent', 'Blauburgunder', 'Blaufränkisch',
           'Scheurebe', 'Ribolla Gialla', 'Charbono',
           'Malbec-Cabernet Sauvignon', 'Pinot Noir-Gamay', 'Pinot Nero',
           'Gros Manseng', 'Nerello Mascalese', 'Shiraz', 'Negroamaro',
           'Champagne Blend', 'Romorantin', 'Syrah-Cabernet Sauvignon',
           'Tannat-Merlot', 'Duras', 'Garnacha', 'Tinta Francisca',
           'Portuguese Sparkling', 'Chenin Blanc', 'Turbiana',
           'Petite Verdot', 'Posip', 'Fumé Blanc', 'Roussanne', 'Grillo',
           'Müller-Thurgau', 'Pinot Auxerrois', 'Port', 'Cabernet Blend',
           'Cabernet Franc-Cabernet Sauvignon', 'Castelão', 'Encruzado',
           'Touriga Nacional-Cabernet Sauvignon', 'Colombard-Sauvignon Blanc',
           'Moscatel', 'Marsanne', 'Siria', 'Garnacha Blanca',
           'Merlot-Cabernet Sauvignon', 'Arinto', 'Petit Manseng', 'Loureiro',
           'Melon', 'Carricante', 'Fiano', 'Schwartzriesling',
           'Sangiovese-Syrah', 'Tannat-Cabernet Franc',
           'Cabernet Franc-Merlot', 'Sauvignon Blanc-Semillon', 'Macabeo',
           'Alfrocheiro', 'Aligoté', 'Verdejo', 'Grenache Blanc',
           'Fernão Pires', 'Spätburgunder', 'Ciliegiolo',
           'Cabernet Sauvignon-Carmenère', 'Auxerrois', 'Sirica', 'Zweigelt',
           'Pugnitello', 'Rosado', 'Rosato', 'Malvazija', 'Kalecik Karasi',
           'Muskat Ottonel', 'Malbec-Bonarda',
           'Tempranillo-Cabernet Sauvignon', 'Rivaner', 'Trepat', 'Baco Noir',
           'Trebbiano', 'Chardonnay-Viognier', 'Syrah-Mourvèdre', 'Graciano',
           'Roviello', 'Perricone', 'Falanghina', 'Vranec', 'Carignan',
           'Cabernet-Shiraz', 'Verdelho', 'Pedro Ximénez',
           'Marsanne-Roussanne', 'Malbec Blend', 'Weissburgunder', 'Morava',
           'Ruen', 'Hondarrabi Zuri', 'Catarratto',
           'Chardonnay-Sauvignon Blanc', 'Vidal', 'Rieslaner', 'Dornfelder',
           'Tinto Fino', 'Gelber Muskateller', 'Roter Veltliner', 'Aragonez',
           'Vitovska', 'Pinot Noir-Syrah', 'Gamay Noir', 'Grauburgunder',
           'Cannonau', 'Mauzac', 'Austrian Red Blend', 'Sémillon',
           'Lambrusco di Sorbara', 'Teran', 'Dolcetto', 'Cinsault',
           'Assyrtico', 'Teroldego', 'Tamjanika', 'Boğazkere', 'Kadarka',
           'Narince', 'Malbec-Petit Verdot', 'Veltliner', 'Traminer',
           'Lambrusco', 'Arneis', 'Cabernet Sauvignon-Shiraz',
           'Tocai Friulano', 'Fer Servadou', 'Muskateller',
           'Nerello Cappuccio', 'Moscatel Roxo', 'Elbling', 'Saperavi',
           'Antão Vaz', 'Pinot Meunier', 'Petite Syrah', 'Malvasia',
           'Malbec-Tannat', 'Kallmet', 'Syrah-Merlot', 'Montepulciano',
           'Kerner', 'Alvarinho-Chardonnay', 'Žilavka', 'Vinhão',
           'Chardonnay-Semillon', 'Carmenère-Cabernet Sauvignon',
           'Merlot-Cabernet Franc', 'Orangetraube',
           'Cabernet Sauvignon-Sangiovese', 'Okuzgozu', 'Viura',
           'Garnacha-Syrah', 'Zibibbo', 'Feteasca', 'Xarel-lo', 'Prokupac',
           'Códega do Larinho', 'Touriga Nacional Blend', 'Inzolia',
           'Cabernet-Syrah', 'Lambrusco Grasparossa', 'Malagousia',
           'Cabernet Franc-Malbec', 'Feteasca Neagra', 'Yapincak',
           'Tempranillo-Shiraz', 'Cabernet Sauvignon Grenache', 'Tinta Roriz',
           'Merlot-Syrah', 'Tinta Fina', 'Colombard-Ugni Blanc', 'Colombard',
           'Roditis', 'Grenache-Carignan', 'Emir', 'Orange Muscat',
           'Karalahna', 'Trincadeira', 'Refosco', 'Pied de Perdrix',
           'Vignoles', 'Carignan-Grenache', "Muscat d'Alexandrie", 'Bobal',
           'Symphony', 'Norton', 'Sauvignon Blanc-Sauvignon Gris',
           'Rkatsiteli', 'Roussanne-Viognier', 'Pinela', 'Blatina',
           'Shiraz-Viognier', 'Bonarda', 'Sauvignon Blanc-Chardonnay',
           'Chambourcin', 'Traminette', 'Grenache Blend', 'Jaen', 'Mondeuse',
           'Feteascǎ Regalǎ', 'Teroldego Rotaliano',
           'Sangiovese-Cabernet Sauvignon', 'Listán Negro',
           'Syrah-Petite Sirah', 'Viognier-Chardonnay', 'Kuntra', 'Jacquère',
           'Portuguiser', 'Grecanico', 'Verdejo-Viura', 'Tinto del Pais',
           'Moscato Giallo', 'Cabernet Sauvignon-Malbec', 'Mission',
           'Neuburger', 'Bastardo', 'Bical', 'Sacy', 'Carineña',
           'Garnacha-Tempranillo', 'Pecorino', 'Garnacha Blend', 'Cococciola',
           'Passerina', 'Gaglioppo', 'Garnacha Tintorera', 'Prieto Picudo',
           'Tempranillo Blanco', "Cesanese d'Affile", 'Muscat Canelli',
           'Cabernet', 'Malvasia Nera', 'Premsal', 'Mansois',
           'Welschriesling', 'Shiraz-Tempranillo', 'Verdicchio', 'Sagrantino',
           'Rolle', 'Trousseau Gris', 'Counoise', 'Mantonico',
           'Cariñena-Garnacha', 'Insolia', 'Tokaji', 'Austrian white blend',
           'Shiraz-Grenache', 'Claret', 'Syrah-Tempranillo', 'Uva di Troia',
           'Aleatico', 'Piedirosso', 'Viognier-Marsanne',
           'Pinot Grigio-Sauvignon Blanc', 'Pallagrello Nero',
           'Chardonnay-Albariño', 'Savagnin', 'Pinotage', 'Braucol',
           'Moschofilero', 'Nero di Troia', 'Carignano', 'Susumaniello',
           'Baga-Touriga Nacional', 'Vidal Blanc', 'Vernaccia',
           'Corvina, Rondinella, Molinara', 'Mavrotragano',
           'Garnacha-Monastrell', 'Lagrein', 'Cabernet Merlot',
           'Monastrell-Syrah', 'Malbec-Tempranillo', 'Syrah-Viognier',
           'Verdeca', 'Sangiovese Grosso', 'Merlot-Argaman',
           'Chenin Blanc-Viognier', 'Garnacha-Cabernet', 'Maturana', 'Malvar',
           'Airen', 'Monica', 'Gewürztraminer-Riesling', 'Prugnolo Gentile',
           'Steen', 'Chenin Blanc-Sauvignon Blanc',
           'Shiraz-Cabernet Sauvignon', 'Picolit', 'Prosecco',
           'White Riesling', 'White Port', 'Zierfandler', 'Petroulianos',
           'Mavrodaphne', 'Savatiano', 'Tempranillo-Garnacha', 'Vidadillo',
           'Syrah-Cabernet', 'Gelber Traminer', 'Grenache-Shiraz',
           'Rotgipfler', 'Cabernet Sauvignon-Tempranillo', 'Edelzwicker',
           'Cortese', 'Chardonnay Weissburgunder', 'Torbato', 'Verduzzo',
           'Debit', 'Bovale', 'Tempranillo-Merlot', 'Xinisteri',
           'Merlot-Cabernet', 'Verdejo-Sauvignon Blanc', 'Black Muscat',
           'Koshu', 'Királyleányka', 'Favorita', 'Xinomavro',
           'Cserszegi Fűszeres', 'Hárslevelü', 'Pallagrello', 'Mavroudi',
           'Muscat Blanc', 'Schiava', 'Meoru', 'Nuragus',
           'Trebbiano di Lugana', 'Coda di Volpe', 'Raboso',
           'Shiraz-Pinotage', 'Enantio', 'Greco Bianco', 'Tai', 'Tokay',
           'Muscadel', 'Cabernet Franc-Carmenère', 'Tintilia ', 'Segalin',
           'Lacrima', 'Cerceal', 'Cayuga', 'Sauvignon Gris', 'Albana',
           'Corvina', 'Macabeo-Moscatel', 'Macabeo-Chardonnay', 'Moscadello',
           'Nasco', 'Viognier-Roussanne', 'Plavac Mali',
           'Cabernet Sauvignon-Merlot-Shiraz', 'Sauvignon Blanc-Chenin Blanc',
           'Shiraz-Mourvèdre', 'Albarín', 'Black Monukka', 'Morio Muskat',
           'Nielluciu', 'Alicante', 'Cabernet Sauvignon and Tinta Roriz',
           'Viura-Chardonnay', "Loin de l'Oeil", 'Roter Traminer',
           'Karasakiz', 'Casavecchia', 'Malvasia-Viura', 'Nosiola',
           'Incrocio Manzoni', 'Viura-Verdejo', 'Erbaluce', 'Forcallà',
           'Pansa Blanca', 'Catalanesca', 'Muscadelle', 'Malbec-Syrah',
           'Petit Meslier', 'Johannisberg Riesling', 'Pignoletto',
           'Cabernet Pfeffer', 'Syrah-Cabernet Franc', 'Valdiguié', 'Mazuelo',
           'Brachetto', 'Jacquez', 'Moscofilero', 'Chardonnay-Sauvignon',
           'Madeleine Angevine', 'Ruché', 'Merlot-Petite Verdot',
           'Roussanne-Marsanne', 'Moscatel de Alejandría',
           'Muscat Blanc à Petit Grain', 'Sämling', 'Mtsvane', 'Zlahtina',
           'Zelen', 'Doña Blanca', 'Carmenère-Syrah',
           'Roussanne-Grenache Blanc', 'Kinali Yapincak', 'Robola',
           'Pinot Blanc-Chardonnay', 'Chardonnay-Pinot Blanc',
           'Saperavi-Merlot', 'Malvasia Istriana', 'Torontel', 'Picapoll',
           'Zierfandler-Rotgipfler', 'Malvasia Fina', 'Chinuri', 'Muscatel',
           'Sousão', 'Silvaner-Traminer', 'Syrah-Carignan', 'Bukettraube',
           'Muskat', 'Argaman', 'Provence white blend', 'Touriga Franca',
           'Morillon', 'Carignan-Syrah', 'Aidani', 'Viognier-Grenache Blanc',
           'Albarossa', 'Sauvignon Blanc-Verdejo', 'Grenache-Mourvèdre',
           'Tannat-Syrah', 'Seyval Blanc', 'Tocai Rosso', 'Pinot-Chardonnay',
           'Moscatel Graúdo', 'Pigato', 'Siegerrebe', 'Bombino Bianco',
           'Trebbiano-Malvasia', 'Magliocco', 'Verduzzo Friulano ',
           'Vespaiolo', 'Marzemino', 'Tempranillo-Malbec', 'Crespiello',
           'Cabernet Franc-Tempranillo', 'Gouveio', 'Caprettone',
           'Garnacha-Graciano', 'Mataro', "Pineau d'Aunis", 'Bual', 'Sercial',
           'Moscato di Noto', 'Sauvignonasse', 'Madeira Blend', 'St. George',
           'Rebula', 'Pallagrello Bianco', 'Vilana', 'Pelaverga Piccolo',
           'Syrah-Grenache-Viognier', 'Alvarelhão', 'Durif', 'Angevine',
           'Semillon-Chardonnay', 'Pinot Blanc-Pinot Noir', 'Manzoni',
           'Maréchal Foch', 'Blauer Portugieser', 'Tocai', 'Shiraz-Malbec',
           'Cabernet Moravia', 'Espadeiro', 'País', 'Altesse', 'Avesso',
           'Grignolino', 'Mandilaria', 'Freisa', 'Merlot-Shiraz', 'Dafni',
           'Xynisteri', 'Grechetto', 'Roscetto', 'Sideritis',
           'Pinotage-Merlot', 'Asprinio', 'Grolleau', 'Gragnano', 'Ansonica',
           'Sangiovese Cabernet', 'Tinta Barroca', 'Syrah-Bonarda',
           'Marsanne-Viognier', 'Azal', 'Durello', 'Syrah-Malbec',
           'Malbec-Cabernet Franc', 'Franconia', 'Rufete', 'Parraleta',
           'St. Vincent', 'Groppello', 'Athiri', 'Muscat of Alexandria',
           'Malvoisie', 'Colorino', 'Merlot-Grenache', 'Terret Blanc',
           'Chardonel', 'Macabeo-Gewürztraminer', 'Grenache Gris', 'Rabigato',
           'Muscat Hamburg', 'Sarba', 'Irsai Oliver', 'Chardonnay-Pinot Gris',
           'Vermentino Nero', 'Pardina', 'Apple', 'Clairette',
           'Sauvignon Musqué', 'Shiraz-Merlot', 'Viognier-Valdiguié',
           'Chardonelle', 'Malmsey', 'Tinta Negra Mole',
           'Pinot Grigio-Chardonnay', 'Muscadet', 'Viura-Sauvignon Blanc',
           'Huxelrebe', 'Tokay Pinot Gris', 'Chardonnay-Pinot Grigio',
           'Moristel', 'Carnelian'], dtype=object)

```python
# find null values 
wine[wine['price'].isnull()].head()
```


```python
# Do some preprocessing to limit the # of wine varities in the dataset
wine = wine[pd.notnull(wine['country'])]
wine = wine[pd.notnull(wine['price'])]
# wine = wine.drop(wine.columns[0], axis=1) 

variety_threshold = 500 # Anything that occurs less than this will be removed.
value_counts = wine['variety'].value_counts()
to_remove = value_counts[value_counts <= variety_threshold].index
wine.replace(to_remove, np.nan, inplace=True)
wine = wine[pd.notnull(wine['variety'])]
```


```python
# Check for Nan cells 
wine.isnull().sum().sum()

nan_rows_price= wine[wine['price'].isnull()].T.any().T

nan_rows_price.count()

```
    0
    
   
```python
# remove all nan prices rows 
newwine=wine[~wine['price'].isnull()]

# remove all nan rating rows 
newwine=newwine[~newwine['points'].isnull()]


```python
# find all the unique points ratings
newwine['points'].unique()
```



    array([ 96,  95,  94,  90,  91,  86,  89,  88,  87,  93,  92,  85,  84,
            83,  82,  81, 100,  99,  98,  97,  80], dtype=int64)


```python
# # get a smaller subset of newwine to reduce memory requirements
newwine=newwine[0:20000]
```


```python
# reduce the records to top 16 varieties  as this counts for 16991 records or 85%

winesub=newwine.loc[(newwine['variety'] == u'Pinot Noir') | (newwine['variety'] == u'Chardonnay') \
   | (newwine['variety'] == u'Cabernet Sauvignon') | (newwine['variety'] == u'Red Blend')  \
   | (newwine['variety'] == u'Bordeaux-style Red Blend') | (newwine['variety'] == u'Sauvignon Blanc') \
   | (newwine['variety'] == u'Riesling') | (newwine['variety'] == u'Syrah')                                   
   | (newwine['variety'] == u'Merlot') | (newwine['variety'] == u'Rosé')
   | (newwine['variety'] == u'Malbec') | (newwine['variety'] == u'Zinfandel') 
   | (newwine['variety'] == u'Nebbiolo') | (newwine['variety'] == u'Portuguese Red')   
   | (newwine['variety'] == u'Sangiovese') | (newwine['variety'] == u'Sparkling Blend')                                         
   | (newwine['variety'] == u'Rhône-style Red Blend') | (newwine['variety'] == u'Cabernet Franc') 
   | (newwine['variety'] == u'Champagne Blend') | (newwine['variety'] == u'Grüner Veltliner') 
   | (newwine['variety'] == u'Pinot Gris') | (newwine['variety'] == u'Gamay')                 
   | (newwine['variety'] == u'Portuguese White') | (newwine['variety'] == u'Viognier')                                          
   | (newwine['variety'] == u'Gewurztraminer') | (newwine['variety'] == u'Pinot Grigio') 
   | (newwine['variety'] == u'Petite Sirah') | (newwine['variety'] == u'Bordeaux-style White Blend') 
   | (newwine['variety'] == u'Carmenère') | (newwine['variety'] == u'Barbera')   
   | (newwine['variety'] == u'Shiraz') | (newwine['variety'] == u'Grenache') 
   | (newwine['variety'] == u'Sangiovese Grosso') | (newwine['variety'] == u'Tempranillo Blend')   
   | (newwine['variety'] == u'Chenin Blanc') ] 
```


```python
# Plot scatter plots
import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(18,12))
ax = fig.gca()

plt.ylim(0, 2500)
plt.xlabel(r'Rating (Points)')
plt.ylabel(r'Price in $')
# plt.legend(loc='upper left')
# actual known points
plt.scatter(x = winesub['points'] ,y=winesub['price'],c='g')
plt.xticks(rotation=90)
plt.show()
```
<img src="output_17_0.png" width="100%">


```python
fig, ax =plt.subplots(1,2)

sns.distplot(winesub['points'],ax=ax[0])
sns.distplot(winesub['price'],ax=ax[1])

fig.show()
```
<img src="output_18_0.png" width="100%">

```python
# Visualise the categorisation 

g = sns.pairplot(data=winesub, hue='variety')
g = (g.set(xlim=(80,101),ylim=(0,2500)))
plt.title("Winery/Brand vs Price")
plt.show(g)

```
<img src="output_19_0.png" width="100%">

```python
# Plot scatter plots
import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(18,12))
ax = fig.gca()

plt.ylim(50, 2500)
plt.xlabel(r'Country')
plt.ylabel(r'Price in $')
plt.legend(loc='upper left')
# actual known points
plt.scatter(x = winesub['country'] ,y=winesub['price'],c='r')
plt.xticks(rotation=45)
plt.show()
```
<img src="output_20_0.png" width="100%">


```python
import matplotlib.pyplot as plt
import seaborn as sns

a4_dims = (16, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)

sns.set_style("whitegrid")
sns.barplot(x="variety", y="price", data=winesub,ax=ax)
# g = sns.lmplot(x="Segment", y="Revenue", data=winesales, aspect=2)

# g = (g.set_axis_labels("Returns","Revenue").set(xlim=(70,101),ylim=(0,120000)))
     
plt.title(" Variety vs Price")
plt.xticks(rotation=90)
 
plt.show(g)
```
<img src="output_21_0.png" width="100%">

```python
import matplotlib.pyplot as plt
import seaborn as sns

a4_dims = (16, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)

sns.set_style("whitegrid")
sns.barplot(x="variety", y="points", data=winesub,ax=ax)
# g = sns.lmplot(x="Segment", y="Revenue", data=winesales, aspect=2)

# g = (g.set_axis_labels("Returns","Revenue").set(xlim=(70,101),ylim=(0,120000)))
     
plt.title(" Variety vs Points")
plt.xticks(rotation=70)
 
plt.show(g)
```
<img src="output_22_0.png" width="100%">


```python
import matplotlib.pyplot as plt
import seaborn as sns

# a4_dims = (11.7, 8.27)
# fig, ax = plt.subplots(figsize=a4_dims)
sns.set_style("whitegrid")
g = sns.lmplot(x="points", y="price", data=winesub, aspect=2,size=5)
g = (g.set_axis_labels("Points","Price").set(xlim=(80,101),ylim=(0,2500)))
plt.title("Points vs Price")
plt.show(g)

```
<img src="output_23_0.png" width="100%">

```python
# Remove all the 5 outliers in price ie > $1000
winetrim=winesub[winesub['price']  <= 1000]
winetrim.describe(include='all')
```

```python
winetrim=winetrim.reset_index(drop=True)
```

```python
import matplotlib.pyplot as plt
import seaborn as sns

# a4_dims = (11.7, 8.27)
# fig, ax = plt.subplots(figsize=a4_dims)
sns.set_style("whitegrid")
g = sns.lmplot(x="points", y="price", data=winetrim, aspect=2,size=5)
g = (g.set_axis_labels("Points","Price").set(xlim=(80,101),ylim=(0,800)))
plt.title("Points vs Price")
plt.show(g)
```
<img src="output_26_0.png" width="100%">


```python
sns.jointplot(x="points", y="price", data=winetrim)
```
<img src="output_27_0.png" width="100%">


```python
# transform Variety data using dummy variables 

winesubext = pd.concat([winetrim, pd.get_dummies(winetrim['variety'],prefix='grapevar_')], axis=1)
```

```python
# set the Varieties as features 

variety=[u'grapevar__Barbera',u'grapevar__Bordeaux-style Red Blend',u'grapevar__Bordeaux-style White Blend',
              u'grapevar__Cabernet Franc',u'grapevar__Cabernet Sauvignon',u'grapevar__Carmenère',u'grapevar__Champagne Blend',
              u'grapevar__Chardonnay',u'grapevar__Chenin Blanc',
              u'grapevar__Grenache',u'grapevar__Grüner Veltliner',u'grapevar__Malbec',u'grapevar__Merlot',u'grapevar__Nebbiolo',
              u'grapevar__Petite Sirah',u'grapevar__Pinot Grigio',u'grapevar__Pinot Gris',u'grapevar__Pinot Noir',
              u'grapevar__Portuguese Red',u'grapevar__Portuguese White',u'grapevar__Red Blend',u'grapevar__Rhône-style Red Blend',
              u'grapevar__Riesling',u'grapevar__Rosé',u'grapevar__Sangiovese',u'grapevar__Sangiovese Grosso',u'grapevar__Sauvignon Blanc',
              u'grapevar__Shiraz',u'grapevar__Sparkling Blend',u'grapevar__Syrah',u'grapevar__Tempranillo Blend',
              u'grapevar__Viognier',u'grapevar__Zinfandel']
winevariety=winesubext[variety]


```python
# Linear regression to determine relationships beteen X and Y 
# choose X and Y 

def lregress (X,y,labelX,labely):

    # split data into test and train and then apply cross validation 

    from sklearn.cross_validation import train_test_split
    # split the data with 50% in each set

    X1, X2, y1, y2 = train_test_split(X, y, random_state=0,
                                      train_size=0.50)
    # OLS regression 

    X = sm.add_constant(X)
    # Note the difference in argument order
    model = sm.OLS(y1, X1).fit()
    # model.summary()
    # predictions = model.predict(X) # make the predictions by the model

    y2_predicted = model.predict(X2)
    model.summary()
    # y2_model
    
    prediction_error = y2 - y2_predicted
    prediction_error

    df1 = pd.DataFrame()
    df1[labely]  = y2
#     df1[labelX] = X2
    df1['predicted y']=y2_predicted
    df1['Residuals'] = df1[labely] -df1['predicted y']
    df1[labely].dropna
    df1['predicted y'].dropna()
    df1['Residuals'].dropna()
    
   
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    %matplotlib inline
  
    sns.distplot(df1['Residuals']) 
    return model.summary() ,sns.lmplot(x=labely, y='predicted y', data=df1)
         
```


```python
# choose X and Y for linear regression eg Points vs Price

nameX=input("X feature eg points: ")
namey = input('y to be predicted eg price: ')

dataX = winetrim[nameX]
datay = winetrim[namey]

lregress(dataX,datay,nameX,namey)
```

    X feature eg points: points
    y to be predicted eg price: price
    

    (<class 'statsmodels.iolib.summary.Summary'>
     """
                                 OLS Regression Results                            
     ==============================================================================
     Dep. Variable:                  price   R-squared:                       0.522
     Model:                            OLS   Adj. R-squared:                  0.522
     Method:                 Least Squares   F-statistic:                 1.004e+04
     Date:                Thu, 17 May 2018   Prob (F-statistic):               0.00
     Time:                        11:21:35   Log-Likelihood:                -45692.
     No. Observations:                9178   AIC:                         9.139e+04
     Df Residuals:                    9177   BIC:                         9.139e+04
     Df Model:                           1                                         
     Covariance Type:            nonrobust                                         
     ==============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
     ------------------------------------------------------------------------------
     points         0.4150      0.004    100.206      0.000       0.407       0.423
     ==============================================================================
     Omnibus:                    12060.438   Durbin-Watson:                   1.988
     Prob(Omnibus):                  0.000   Jarque-Bera (JB):          3595367.884
     Skew:                           7.256   Prob(JB):                         0.00
     Kurtosis:                      98.870   Cond. No.                         1.00
     ==============================================================================
     
     Warnings:
     [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
     """, <seaborn.axisgrid.FacetGrid at 0x2265a5afda0>)

<img src="output_31_2.png" width="100%">


<img src="output_31_3.png" width="100%">

```python
# setup tokennizer NLP for wine reviews description 
from collections import defaultdict
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
import scipy as sp
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
%matplotlib inline
# define X and y
XX = winetrim['description'].values
yy = winetrim['points'].values
zz = winetrim['price'].values
```


```python
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

custom_stop_words = list(ENGLISH_STOP_WORDS)

# You can of course add your own custom stopwords
custom_stop_words.append('wine')
custom_stop_words.append('bordeaux')
```


```python
# use the count vectorizer to compare results 
# include 1-grams and 2-grams
vect = CountVectorizer(ngram_range=(1, 5),max_features=500,stop_words='english')
X_train_dtm = vect.fit_transform(XX)
countfeatures=vect.get_feature_names()
X_train_dtm.shape
len(countfeatures)
# # last 50 features
print (countfeatures[-50:])
```


    ['tight', 'tightly', 'time', 'toast', 'toasted', 'toasty', 'tobacco', 'tomato', 'tones', 'touch', 'touches', 'tropical', 'tropical fruit', 'valley', 'value', 'vanilla', 'varietal', 'variety', 'velvety', 'verdot', 'vibrant', 'vines', 'vineyard', 'vineyards', 'vintage', 'violet', 'warm', 'way', 'weight', 'wet', 'whiff', 'white', 'white pepper', 'wild', 'wine', 'wine offers', 'wine shows', 'winemaker', 'winery', 'wines', 'wood', 'wood aging', 'wrapped', 'year', 'years', 'yellow', 'young', 'zest', 'zesty', 'zinfandel']
    


```python
# # create a DF based on the combined vectors outputs from the count vectorizer
dfwinerate2=pd.DataFrame(data=X_train_dtm.todense(),columns=countfeatures)
# dfwinerate2.head(4)
```


```python
# Run linear regression of words vectoriser scores vs Points 
# choose X and Y for linear regression 

nameX1=input("X feature eg bag of words from vectorizer: ")
namey1 = input('y to be predicted eg points : ')

dataX1 = dfwinerate2.values
datay1 = yy

lregress(dataX1,datay1,nameX1,namey1)

```

    X feature eg bag of words from vectorizer: words
    y to be predicted eg points : points
    




    (<class 'statsmodels.iolib.summary.Summary'>
     """
                                 OLS Regression Results                            
     ==============================================================================
     Dep. Variable:                      y   R-squared:                       0.955
     Model:                            OLS   Adj. R-squared:                  0.952
     Method:                 Least Squares   F-statistic:                     366.3
     Date:                Thu, 17 May 2018   Prob (F-statistic):               0.00
     Time:                        11:23:50   Log-Likelihood:                -39971.
     No. Observations:                9178   AIC:                         8.094e+04
     Df Residuals:                    8678   BIC:                         8.451e+04
     Df Model:                         500                                         
     Covariance Type:            nonrobust                                         
     ==============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
     ------------------------------------------------------------------------------
     x1             0.4559      1.594      0.286      0.775      -2.669       3.581
     x2             3.0031      1.664      1.805      0.071      -0.258       6.264
     x3            -1.2236      1.804     -0.678      0.498      -4.759       2.312
     x4             3.4445      1.822      1.891      0.059      -0.126       7.015
     :
     x499          -3.8866      2.446     -1.589      0.112      -8.681       0.908
     x500           0.6928      3.225      0.215      0.830      -5.630       7.016
     ==============================================================================
     Omnibus:                    11900.425   Durbin-Watson:                   1.993
     Prob(Omnibus):                  0.000   Jarque-Bera (JB):          4088433.438
     Skew:                           7.000   Prob(JB):                         0.00
     Kurtosis:                     105.445   Cond. No.                         318.
     ==============================================================================
     
     Warnings:
     [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
     """, <seaborn.axisgrid.FacetGrid at 0x2265f7f0a20>)


<img src="output_37_2.png" width="100%">


<img src="output_37_3.png" width="100%">


```python
# set up OVR logit regression for wine varieties using the word counts 

def logregress(grape_variety):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # split data into Train and Test for  X and y_
    X = dfwinerate2.values  # array of 16345 rows x 500 words 
    # y = winevariety[u'grapevar__Barbera'] #array of  16345 rows x 1 for points 
    y = winevariety[grape_variety].values #array of  16345 rows x 1 for points 

    # split the new DataFrame into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1,train_size=0.5)

    # Run Multinomial  Logit regression to check if description  terms can be used to predict variety  

    model=LogisticRegression(multi_class='ovr',solver ='newton-cg').fit(X_train,y_train)
    # print model.summary()
    yhat = model.predict(X_test)  # will output array with integer values.

#     print (yhat, model.coef_,model.score, accuracy_score(y_test, yhat))
    
    # change the array into a list 
    import numpy as np
    modelcoef=np.array(model.coef_).tolist()
    len(modelcoef)
    len(countfeatures)
    
    # split the list into individual items
    import itertools
    words = list(itertools.chain(*modelcoef))
#     words
    
#     setup a dictionary with word counts and word labels
    varietydict = dict(zip(countfeatures,words))
#     varietydict

# add accuracy score and wine variety to dictionary
    varietydict['0Accuracy'] = accuracy_score(y_test, yhat)*100
   
#     varietydict
# sort the words with largest word counts first 
    sorted_d = sorted(varietydict.items(), key=lambda x: x[1],reverse=True)
#     sorted_d
    varietydict['0Grape'] = grape_variety

    # Calculate the confusion matrix metrics for your model below.¶
    from sklearn.metrics import classification_report
    print(grape_variety,'\n',classification_report(y_test, yhat))
    

#     get top 50 words scores 
    results=[grape_variety]+sorted_d[:500]
    return results
```


```python
# Run logit regression of words vectoriser scores vs variety of grapes for all grapes variety

# choose X and Y for linear regression 
dfwinefinal= pd.DataFrame()

for w in winevariety:
    #     print(w)
    # grapename = input('enter grape variety:')
#     ugrape='grapevar__'+grapename
    ugrape=w

    dfwinefinal[ugrape]=logregress(ugrape)
dfwinefinal
```

    C:\Users\bless\Anaconda3\lib\site-packages\sklearn\model_selection\_split.py:2026: FutureWarning: From version 0.21, test_size will always complement train_size unless both are specified.
      FutureWarning)
    

    grapevar__Barbera 
                  precision    recall  f1-score   support
    
              0       0.99      1.00      1.00      9119
              1       0.90      0.15      0.26        60
    
    avg / total       0.99      0.99      0.99      9179
    
    grapevar__Bordeaux-style Red Blend 
                  precision    recall  f1-score   support
    
              0       0.97      0.99      0.98      8712
              1       0.63      0.40      0.49       467
    
    avg / total       0.95      0.96      0.95      9179
    
    grapevar__Bordeaux-style White Blend 
                  precision    recall  f1-score   support
    
              0       0.99      1.00      1.00      9117
              1       0.20      0.03      0.06        62
    
    avg / total       0.99      0.99      0.99      9179
    
    grapevar__Cabernet Franc 
                  precision    recall  f1-score   support
    
              0       0.99      1.00      0.99      9077
              1       0.57      0.08      0.14       102
    
    avg / total       0.99      0.99      0.98      9179
    
    grapevar__Cabernet Sauvignon 
                  precision    recall  f1-score   support
    
              0       0.95      0.98      0.96      8304
              1       0.73      0.48      0.58       875
    
    avg / total       0.93      0.93      0.93      9179
    
    grapevar__Carmenère 
                  precision    recall  f1-score   support
    
              0       1.00      1.00      1.00      9135
              1       0.47      0.16      0.24        44
    
    avg / total       0.99      1.00      0.99      9179
    
    

    C:\Users\bless\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1135: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    

    grapevar__Champagne Blend 
                  precision    recall  f1-score   support
    
              0       1.00      1.00      1.00      9139
              1       0.00      0.00      0.00        40
    
    avg / total       0.99      1.00      0.99      9179
    
    grapevar__Chardonnay 
                  precision    recall  f1-score   support
    
              0       0.95      0.98      0.97      8157
              1       0.78      0.63      0.70      1022
    
    avg / total       0.93      0.94      0.94      9179
    
    grapevar__Chenin Blanc 
                  precision    recall  f1-score   support
    
              0       0.99      1.00      1.00      9124
              1       0.44      0.07      0.12        55
    
    avg / total       0.99      0.99      0.99      9179
    
    grapevar__Grenache 
                  precision    recall  f1-score   support
    
              0       0.99      1.00      1.00      9114
              1       0.80      0.12      0.21        65
    
    avg / total       0.99      0.99      0.99      9179
    
    grapevar__Grüner Veltliner 
                  precision    recall  f1-score   support
    
              0       0.99      1.00      1.00      9087
              1       0.56      0.21      0.30        92
    
    avg / total       0.99      0.99      0.99      9179
    
    grapevar__Malbec 
                  precision    recall  f1-score   support
    
              0       0.98      0.99      0.99      8899
              1       0.67      0.34      0.45       280
    
    avg / total       0.97      0.97      0.97      9179
    
    grapevar__Merlot 
                  precision    recall  f1-score   support
    
              0       0.98      0.99      0.99      8909
              1       0.69      0.40      0.50       270
    
    avg / total       0.97      0.98      0.97      9179
    
    grapevar__Nebbiolo 
                  precision    recall  f1-score   support
    
              0       0.99      1.00      0.99      9028
              1       0.75      0.54      0.63       151
    
    avg / total       0.99      0.99      0.99      9179
    
    grapevar__Petite Sirah 
                  precision    recall  f1-score   support
    
              0       0.99      1.00      1.00      9116
              1       0.53      0.13      0.21        63
    
    avg / total       0.99      0.99      0.99      9179
    
    grapevar__Pinot Grigio 
                  precision    recall  f1-score   support
    
              0       0.99      1.00      1.00      9121
              1       0.69      0.16      0.25        58
    
    avg / total       0.99      0.99      0.99      9179
    
    grapevar__Pinot Gris 
                  precision    recall  f1-score   support
    
              0       0.99      1.00      0.99      9041
              1       0.52      0.16      0.24       138
    
    avg / total       0.98      0.99      0.98      9179
    
    grapevar__Pinot Noir 
                  precision    recall  f1-score   support
    
              0       0.93      0.97      0.95      7821
              1       0.80      0.60      0.69      1358
    
    avg / total       0.91      0.92      0.91      9179
    
    grapevar__Portuguese Red 
                  precision    recall  f1-score   support
    
              0       0.99      1.00      0.99      9002
              1       0.58      0.28      0.37       177
    
    avg / total       0.98      0.98      0.98      9179
    
    grapevar__Portuguese White 
                  precision    recall  f1-score   support
    
              0       0.99      1.00      1.00      9120
              1       0.19      0.07      0.10        59
    
    avg / total       0.99      0.99      0.99      9179
    
    grapevar__Red Blend 
                  precision    recall  f1-score   support
    
              0       0.95      0.98      0.96      8375
              1       0.67      0.44      0.53       804
    
    avg / total       0.92      0.93      0.93      9179
    
    grapevar__Rhône-style Red Blend 
                  precision    recall  f1-score   support
    
              0       0.99      1.00      0.99      9038
              1       0.56      0.21      0.30       141
    
    avg / total       0.98      0.99      0.98      9179
    
    grapevar__Riesling 
                  precision    recall  f1-score   support
    
              0       0.98      0.99      0.99      8647
              1       0.85      0.69      0.76       532
    
    avg / total       0.97      0.98      0.97      9179
    
    grapevar__Rosé 
                  precision    recall  f1-score   support
    
              0       0.98      1.00      0.99      8842
              1       0.82      0.53      0.64       337
    
    avg / total       0.98      0.98      0.98      9179
    
    grapevar__Sangiovese 
                  precision    recall  f1-score   support
    
              0       0.98      0.99      0.99      8943
              1       0.57      0.37      0.45       236
    
    avg / total       0.97      0.98      0.97      9179
    
    grapevar__Sangiovese Grosso 
                  precision    recall  f1-score   support
    
              0       0.99      1.00      1.00      9046
              1       0.82      0.57      0.67       133
    
    avg / total       0.99      0.99      0.99      9179
    
    grapevar__Sauvignon Blanc 
                  precision    recall  f1-score   support
    
              0       0.97      0.99      0.98      8723
              1       0.70      0.41      0.52       456
    
    avg / total       0.96      0.96      0.96      9179
    
    grapevar__Shiraz 
                  precision    recall  f1-score   support
    
              0       0.99      1.00      1.00      9095
              1       0.58      0.26      0.36        84
    
    avg / total       0.99      0.99      0.99      9179
    
    grapevar__Sparkling Blend 
                  precision    recall  f1-score   support
    
              0       0.99      1.00      0.99      9059
              1       0.48      0.09      0.15       120
    
    avg / total       0.98      0.99      0.98      9179
    
    grapevar__Syrah 
                  precision    recall  f1-score   support
    
              0       0.97      0.99      0.98      8715
              1       0.75      0.38      0.50       464
    
    avg / total       0.96      0.96      0.96      9179
    
    grapevar__Tempranillo Blend 
                  precision    recall  f1-score   support
    
              0       0.99      1.00      1.00      9104
              1       0.61      0.15      0.24        75
    
    avg / total       0.99      0.99      0.99      9179
    
    grapevar__Viognier 
                  precision    recall  f1-score   support
    
              0       0.99      1.00      0.99      9071
              1       0.33      0.04      0.07       108
    
    avg / total       0.98      0.99      0.98      9179
    
    grapevar__Zinfandel 
                  precision    recall  f1-score   support
    
              0       0.98      1.00      0.99      8928
              1       0.64      0.28      0.39       251
    
    avg / total       0.97      0.98      0.97      9179
    
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>grapevar__Barbera</th>
      <th>grapevar__Bordeaux-style Red Blend</th>
      <th>grapevar__Bordeaux-style White Blend</th>
      <th>grapevar__Cabernet Franc</th>
      <th>grapevar__Cabernet Sauvignon</th>
      <th>grapevar__Carmenère</th>
      <th>grapevar__Champagne Blend</th>
      <th>grapevar__Chardonnay</th>
      <th>grapevar__Chenin Blanc</th>
      <th>grapevar__Grenache</th>
      <th>grapevar__Grüner Veltliner</th>
      <th>grapevar__Malbec</th>
      <th>grapevar__Merlot</th>
      <th>grapevar__Nebbiolo</th>
      <th>grapevar__Petite Sirah</th>
      <th>grapevar__Pinot Grigio</th>
      <th>grapevar__Pinot Gris</th>
      <th>grapevar__Pinot Noir</th>
      <th>grapevar__Portuguese Red</th>
      <th>grapevar__Portuguese White</th>
      <th>grapevar__Red Blend</th>
      <th>grapevar__Rhône-style Red Blend</th>
      <th>grapevar__Riesling</th>
      <th>grapevar__Rosé</th>
      <th>grapevar__Sangiovese</th>
      <th>grapevar__Sangiovese Grosso</th>
      <th>grapevar__Sauvignon Blanc</th>
      <th>grapevar__Shiraz</th>
      <th>grapevar__Sparkling Blend</th>
      <th>grapevar__Syrah</th>
      <th>grapevar__Tempranillo Blend</th>
      <th>grapevar__Viognier</th>
      <th>grapevar__Zinfandel</th>
    </tr>
  </thead>
