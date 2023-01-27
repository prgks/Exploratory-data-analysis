#!/usr/bin/env python
# coding: utf-8

# # Материалы:
# 
# * [Презентация]()
# * [Дашборд]()

# 
# # Декомпозиция проекта:
# 
#  1. [Введение:](#p1)
#       * [Описание проекта](#p1.1)  
#       * [Задачи проекта](#p1.2)
#       * [Описание данных](#p1.3)
#       
#  
#  2. [Загрузка и изучение данных:](#p2)
#       * [Загрузка данных](#p2.1)  
#       * [Изучение данных](#p2.2)
#       * [Вывод](#p2.3)
#    
#    
#  3. [Предобработка данных:](#p3)
#       * [Приведение названия столбцов к стандартному виду](#p3.1)  
#       * [Преобразование данных к необходимому для анализа типу](#p3.2)
#       * [Проверка данных на явные и неявные дубликаты](#p3.3)
#       * [Проверка данных на пропуски и аномалии](#p3.4)
#       * [Объединение таблиц](#p3.5)
#       * [Вывод](#p3.6)
#    
#    
#  4. [Исследовательский анализ данных:](#p4)
#       * [Retention rate](#p4.1)  
#       * [Время, проведенное в преложении пользователями](#p4.2)
#       * [Частота действий пользователей](#p4.3)
#       * [Конверсия в целевое действие(просмотр контактов)](#p4.4)
#       * [Вывод](#p4.5)
#    
#    
#  5. [Сегментация пользователей на основе действий:](#p5)
#       * [Выбор метода сегментации пользователей](#p5.1)  
#       * [Выделение групп пользователей](#p5.2)
#       * [Вывод](#p5.3)
#     
#     
#  6. [Ответы на вопросы заказчика в разрезе выделенных групп:](#p6)
#       * [Retention rate](#p6.1)  
#       * [Конверсия в целевое действие](#p6.2)
#       * [Вывод](#p6.3)
#       
#    
#  7. [Проверка статистических гипотиз:](#p7)
#       * [Гипотеза 1: Конверсии групп пользователей,  установивших приложение по ссылкам из yandex и  google, в contacts_show равны.](#p7.1)  
#       * [Гипотеза 2:  Конверсии групп пользователей, использующих приложение днем и  в утро/ночь, в contacts_call равны.](#p7.2)
#       * [Гипотеза 3:  Конверсии групп пользователей, мультиинтервал с каждой из других групп, в tips_click.](#p7.3)
#       * [Вывод](#p7.4)
# 
# 
#  8. [Общий вывод:](#p8)
#     * [Общий вывод:](#p8.1)
#     * [Рекомендации](#p8.2)

# <a id="p1"></a>
# ## Введение

# <a id="p1.1"></a>
# ### Описание проекта
# 

#    В рамках данного проекта необходимо провести исследовательский анализ данных, содержащих информацию о событиях, совершенных пользователями мобильного приложения "Ненужные вещи", на основании ключевых метрик произвести сегментацию пользователей, с последующим ответом на вопросы заказчика в разрезе выделенных групп. Так же предстоит проверить статистические гипотизы, по результатам которых, будут даны рекомендации.

# <a id="p1.2"></a>
# ### Задачи проекта
# * Выполнить исследовательский анализ данных
# * Произвести сегментацию пользователей
# * Ответить на вопросы заказчика
# * Проверить статистические гипотизы
# * По результатам исследования дать рекомендации
# * Подготовить презентацию и дашборд

# <a id="p1.3"></a>
# ### Описание данных

# Датасет содержит данные о событиях, совершенных в мобильном приложении "Ненужные вещи".
# 
# Датасет mobile_dataset.csv содержит колонки:
# * event.time — время совершения
# * event.name — название события
# * user.id — идентификатор пользователя
# 
# 
# Датасет mobile_sources.csv содержит колонки:
# * userId — идентификатор пользователя
# * source — источник, с которого пользователь установил приложение
# 
# Расшифровки событий:
# * advert_open — открытие карточки объявления
# * photos_show — просмотр фотографий в объявлении
# * tips_show — пользователь увидел рекомендованные объявления
# * tips_click — пользователь кликнул по рекомендованному объявлению
# * contacts_show и show_contacts — пользователь нажал на кнопку "посмотреть номер телефона" на карточке объявления
# * contacts_call — пользователь позвонил по номеру телефона на карточке объявления
# * map — пользователь открыл карту размещенных объявлений
# * search_1 — search_7 — разные события, связанные с поиском по сайту
# * favorites_add — добавление объявления в избранное

#  

# <a id="p2"></a>
# ## Загрузка и изучение данных

# <a id="p2.1"></a>
# ### Загрузка данных

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px 
import numpy as np
from datetime import datetime, timedelta
import math
import re
import scipy.stats as st


# In[2]:


try:
    soures = pd.read_csv('https://code.s3.yandex.net/datasets/mobile_soures.csv')
except:
    soures = pd.read_csv('mobile_soures.csv')
    
try:
    dataset = pd.read_csv('https://code.s3.yandex.net/datasets/mobile_dataset.csv')
except:
    dataset = pd.read_csv('mobile_dataset.csv')


# <a id="p2.2"></a>
# ### Изучение данных

# In[3]:


soures.describe()


# В таблице soures 4293 строки, столько же уникальных Id пользователей и 3 типа источников привлечения пользователей, из которых yandex является самым популярным - 1934 пользователя.

# In[4]:


dataset.info()


# Колонку со временем необходимо будет привести к временному формату

# In[5]:


dataset.describe()


# В таблице dataset 74197 строк, 16 уникальных событий, самое популярное из которых -  tips_show(40055), 4293 уникальных Id пользователей, наибольшее количество среди действий одного пользователя - 478

# <a id="p2.3"></a>
# ### Вывод

# На данном шаге были загружены данные и сделаны следующие выводы:
# 
# 1) В таблице soures 4293 строки, столько же уникальных Id пользователей и 3 типа источников привлечения пользователей, из которых yandex является самым популярным - 1934 пользователя.
# 
# 2) В таблице dataset 74197 строк, 16 уникальных событий, самое популярное из которых -  tips_show(40055), 4293 уникальных Id пользователей, наибольшее количество среди действий одного пользователя - 478

#  

# <a id="p3"></a>
# ## Предобработка данных

# <a id="p3.1"></a>
# ### Приведение названия столбцов к стандартному виду

# In[6]:


soures.rename(columns = {'userId' : 'user_id'}, inplace = True)
dataset.columns = [x.replace('.', '_') for x in dataset.columns]


# <a id="p3.2"></a>
# ### Преобразование данных к необходимому для анализа типу

# In[7]:


dataset['event_time'] = pd.to_datetime(dataset['event_time'])


# <a id="p3.3"></a>
# ### Проверка данных на явные и неявные дубликаты

# In[8]:


dataset.groupby('event_name',as_index = False).agg({'user_id':'count'}).sort_values('user_id')


# По условию contacts_show и show_contacts обозначают одно и тоже действие, переименуем show_contacts	 в  contacts_show, так же выделим search  в одну группу, а затем проверим на дубликаты

# In[9]:


dataset['event_name'] = dataset['event_name'].apply(lambda x: 'contacts_show' if x == 'show_contacts' else x)


# In[10]:


dataset['event_name'] = dataset['event_name'].apply(lambda x: 'search' if re.search('search', str(x)) else x)


# In[11]:


dataset.groupby('event_name',as_index = False).agg({'user_id':'count'}).sort_values('user_id')


# In[12]:


print(dataset.duplicated().sum())      
print(soures.duplicated().sum())


# Округлим время в данных до секунд и еще раз проверим на дубликаты

# In[13]:


df = dataset.copy()
dataset['event_time'] = dataset['event_time'].astype('datetime64[s]')


# In[14]:


dataset.duplicated().sum()


# In[15]:


round(dataset.duplicated().sum()/dataset['user_id'].count()*100,2)


# Во втором случае обнаружено 1143 дубликата, что составляет 1.5 процента от всех данных.
# Изучим их подробнее.

# In[16]:


duplicate = dataset[dataset.duplicated() == True].sort_values(['user_id','event_time'])
duplicate.groupby('event_name',as_index = False).agg({'user_id':'count'}).sort_values('user_id', ascending = False)


# Большинство дубликатов образовалось при действии photos_show, в теории это возможно если очень быстро листать фото.
# Посмотрим разницу по времени между дубликатами

# In[17]:


listt = list(duplicate['user_id'].unique())


# In[18]:


df[(df['user_id'].isin(listt))&(df['event_name'] == 'photos_show')].sort_values(['user_id','event_time']).head(50)


# В среднем разница между действиями photos_show наших дубликатов около 0.1 секунды. Если учесть малую вероятность данных действий и количество дубликатов - всего 1.5 процента, для чистоты исследования эти данные лучше удалить

# In[19]:


dataset = dataset.drop_duplicates().reset_index(drop=True)


# <a id="p3.4"></a>
# ### Проверка данных на пропуски и аномалии

# In[20]:


print(dataset.isna().sum())
print(soures.isna().sum())


# In[21]:


dataset['event_time'].min()


# In[22]:


dataset['event_time'].max()


# In[23]:


(dataset['event_time'].max() - dataset['event_time'].min())


# Представлены данные за 28 дней с 7 октября 2019 по 3 ноября 2019

# In[24]:


dataset.hist(figsize = (20,10), bins = 28)
plt.title('Распределение событий по дням');


# In[25]:


dataset.hist(figsize = (20,10), bins = 112)
plt.title('Распределение событий по времени суток');


#     По дням данные предтавлены  не равномерно с возрастаниями и просадками(возможно обусловлено привязкой ко дням недели и выходным, так же можно выделить относительно небольшое количество данных 3000 событий за самый активный день). В течении дня количество стабильно растет с ночи и падает к вечеру, за исключением некоторых дней, в которых в течении всего времени возрастает(такие дни преимущественно субботы).

# <a id="p3.5"></a>
# ### Объединение таблиц

# In[26]:


merge_data = dataset.merge(soures, on='user_id', how='left')


# Проверим объединенную таблицу на выбивающиеся значения действий на одного пользователя

# In[27]:


df = merge_data.groupby('user_id',as_index = False).agg({'event_time':'count'})
np.percentile(df['event_time'],[95,99])


# Только 1 процент пользователей совершает больше чем 128 действий 

# In[28]:


np.percentile(df['event_time'],[1,5])


# только 1 процент пользователей делает только 1 действие. Cоздадим дополнительный датафрейм, удалив эти 2 процента пользователей

# In[29]:


lists = list(df[(df['event_time'] > 128)|(df['event_time'] == 1)]['user_id'])
merge_data_filt = merge_data[~(merge_data['user_id'].isin(lists))].reset_index(drop = True)
merge_data_filt


# Удалили около 15 процентов от всех событий

# In[30]:


merge_data_filt.hist(figsize = (20,10), bins = 28)
plt.title('Распределение событий по дням');


# In[31]:


merge_data_filt.hist(figsize = (20,10), bins = 112)
plt.title('Распределение событий по времени суток');


# Распределение не изменилось, в дальнейшем при необходимости будем проверять два типа таблиц(с фильтром и без)

# <a id="p3.6"></a>
# ### Вывод

# 1) На данном этапе названия столбцов были приведены к стандартному виду, данные так же приведены к нужному для последующей работы типу, а также проверены на дубликаты и пропуски.
# 
# 2) Были объеденены 2 таблицы для последующего анализа и изучено распределение данных по времени:
# 
# По дням данные предтавлены с небольшими возрастаниями и просадками, так же в течении дня количество стабильно растет с ночи и падает к вечеру, за исключением некоторых дней, в которых в течении всего времени возрастает(такие дни преимущественно субботы).

#  

# <a id="p4"></a>
# ## Исследовательский анализ данных

# <a id="p4.1"></a>
# ### Retention rate

# создадим таблицу с первый появлением пользователей

# In[32]:


def get_profiles(sessions):
    frame = (
    sessions.sort_values(by=['user_id', 'event_time'])
    .groupby('user_id')
    .agg({'event_time': 'first', 'source': 'first'})
    .rename(columns={'event_time': 'first_ses'})
    .reset_index())
    frame['dt'] = frame['first_ses'].dt.date
    frame['month'] = frame['first_ses'].astype('datetime64[M]')
    return frame


# In[33]:


frame =  get_profiles(merge_data)
frame_filt = get_profiles(merge_data_filt)


# Зададим функцию для определения Ratation rite, и для визуализации ее с помощью heatmap

# In[34]:


def get_retention(
    frame, sessions, observation_date, horizon_days, dimensions=[], ignore_horizon=False
):

    last_suitable_acquisition_date = observation_date
    if not ignore_horizon:
        last_suitable_acquisition_date = observation_date - timedelta(
            days=horizon_days - 1
        )
    result_raw = frame.query('dt <= @last_suitable_acquisition_date')

    result_raw = result_raw.merge(
        sessions[['user_id', 'event_time']], on='user_id', how='left'
    )
    result_raw['lifetime'] = (
        result_raw['event_time'] - result_raw['first_ses']
    ).dt.days
        
    result_grouped = result_raw.pivot_table(
        index= dimensions, columns='lifetime', values='user_id', aggfunc='nunique'
    )
    cohort_sizes = (
        result_raw.groupby(dimensions)
        .agg({'user_id': 'nunique'})
        .rename(columns={'user_id': 'cohort_size'})
    )
    result_grouped = cohort_sizes.merge(
        result_grouped, on=dimensions, how='left'
    ).fillna(0)
    result_grouped = result_grouped.div(result_grouped['cohort_size'], axis=0)

    result_grouped = result_grouped[
        ['cohort_size'] + list(range(horizon_days))
    ]

    result_grouped['cohort_size'] = cohort_sizes

    return result_raw, result_grouped


# In[35]:


def heatmap(retention, name = 'Тепловая карта'):
    plt.figure(figsize=(15, 6))
    sns.heatmap(retention.drop(columns=['cohort_size', 0]), annot=True, fmt='.2%')
    plt.title(name)
    return plt.show()


# Изучим удержание пользователей по дням

# In[36]:


observation_date = datetime(2019, 11, 3).date()
horizon_days = 14
result_raw, result_grouped = get_retention(
    frame, merge_data, observation_date, horizon_days, dimensions = ['dt'], ignore_horizon=False
)
heatmap(result_grouped, name = 'Тепловая карта удержания по дням')


# Ratation rate далек от идельного, нет планомерного постоянного снижения, временами очень сильно проседает по дням, а затем идет опять в рост(это может быть обусловлено 2 факторами - малым количеством количеством пользователей в когортах по дням,  специфика приложения неподрозумевает его ежедневное пользование, поэтому если пользователь не заходит день или два, это не означает, что он забросил приложение), Проглядывается снижение удержания к 14 дню групп с 19 октября,  а так же высокий показатель RR(ratation rate) группы 14 октября и очень хороший старт у когорт 12 и 17 октября

# Проверим отфильтрованные данные

# In[37]:


observation_date = datetime(2019, 11, 3).date()
horizon_days = 14
result_raw, result_grouped = get_retention(
    frame_filt, merge_data_filt, observation_date, horizon_days, dimensions = ['dt'], ignore_horizon=False
)
heatmap(result_grouped, name = 'Тепловая карта удержания по дням')


# В целом ситуация особо не изменилось проверим, проверим RR  по неделям. ДЛя это создадим отдельную маркер для недели, когда пользователь начал пользоваться приложением

# In[38]:


def week(x):
    if x < pd.Timestamp(2019, 10, 14):
        return 1
    
    elif  pd.Timestamp(2019, 10, 14) <= x < pd.Timestamp(2019, 10, 21):
        return 2
    
    elif  pd.Timestamp(2019, 10, 21) <= x < pd.Timestamp(2019, 10, 28):
        return 3
    else:
        return 4
    
           


# In[39]:


frame['week']  = frame['first_ses'].apply(week)
frame_filt['week']  = frame_filt['first_ses'].apply(week)


# In[99]:


frame


# Дополним функцию для разбивки по неделям, а так же для разбивки без групп

# In[40]:


def get_retention(
    frame, sessions, observation_date, horizon_days, dimensions=[], ignore_horizon=False, week_max = 4
):

    last_suitable_acquisition_date = observation_date
    if not ignore_horizon:
        last_suitable_acquisition_date = observation_date - timedelta(
            days=horizon_days - 1
        )
    result_raw = frame.query('dt <= @last_suitable_acquisition_date')
    if 'week' in dimensions:
        weeks = math.ceil(horizon_days/7)
        last_week = week_max - weeks
        result_raw = result_raw.query('week <= @last_week')
        
    result_raw = result_raw.merge(
        sessions[['user_id', 'event_time']], on='user_id', how='left'
    )
    result_raw['lifetime'] = (
        result_raw['event_time'] - result_raw['first_ses']
    ).dt.days
    
    if len(dimensions) == 0:
        
        result_raw['cohort'] = 'All users'
       
        dimensions = dimensions + ['cohort']
    
    result_grouped = result_raw.pivot_table(
        index= dimensions, columns='lifetime', values='user_id', aggfunc='nunique'
    )
    cohort_sizes = (
        result_raw.groupby(dimensions)
        .agg({'user_id': 'nunique'})
        .rename(columns={'user_id': 'cohort_size'})
    )
    result_grouped = cohort_sizes.merge(
        result_grouped, on=dimensions, how='left'
        ).fillna(0)
    result_grouped = result_grouped.div(result_grouped['cohort_size'], axis=0)

    result_grouped = result_grouped[
        ['cohort_size'] + list(range(horizon_days))
    ]

    result_grouped['cohort_size'] = cohort_sizes

    return result_raw, result_grouped


# Зададим конечную дату исследования и горизонт анализа

# In[41]:


observation_date = datetime(2019, 11, 3).date()
horizon_days = 14


# In[42]:


result_raw, result_grouped = get_retention(
    frame, merge_data, observation_date, horizon_days, dimensions = ['week'], ignore_horizon= False
)
heatmap(result_grouped, name = 'Тепловая карта удержания по неделям')


# Первая неделя показывает лучший RR к 14 дню, и приблизительно равный к 7, так же удержание стало заметно ровнее

# Посмотрим разбивку по источникам привлечения

# In[43]:


result_raw, result_grouped = get_retention(
    frame, merge_data, observation_date, horizon_days, dimensions = ['source'], ignore_horizon= False
)
heatmap(result_grouped, name = 'Тепловая карта удержания по источникам')


# Создадим функцию для визуализации c помощью графика

# In[44]:


def graf(retention, name = 'удержания'):
    report = retention.drop(columns = ['cohort_size', 0])

    report.T.plot(
    grid=True,
    xticks=list(report.columns.values),
    figsize=(15, 10)
    )
    plt.xlabel('Лайфтайм')
    plt.title(name) 
    return plt.show()


# In[45]:


graf(result_grouped, name = 'Кривые удержания по источникам')


# К концу 14 дня у других источников удержание лучше, но нельзя однозначно сказать что они лучше на всем отрезке, графики очень неравномерные, лидеры все время меняются

# Рассмотри общее удержание пользователей

# In[46]:


result_raw, result_grouped = get_retention(
    frame, merge_data, observation_date, horizon_days, dimensions = [], ignore_horizon= False
)
graf(result_grouped, name = 'Кривая удержания для всех пользователей')


# К концу 1 недели удержание  6 процентов, к концу второй 4

# Рассмотри удержание пользователей по неделям и источникам

# In[47]:


result_raw, result_grouped = get_retention(
    frame, merge_data, observation_date, horizon_days, dimensions = ['source', 'week'], ignore_horizon= False
)
graf(result_grouped, name = 'Кривые удержания по источникам и неделям')


# Так же нельзя выделить никакую из групп, данные очень не равномерны, попробуем посмотреть еще раз использую отфильрованные данные, а также же в горизонте исследования 7 дней

# In[48]:


result_raw, result_grouped = get_retention(
    frame_filt, merge_data_filt, observation_date, horizon_days, dimensions = ['source','week'], ignore_horizon= False
)
graf(result_grouped, name = 'Кривые удержания по источникам и неделям')


# Однозначных выводов все так же нельзя сделать, учитывая что мы увеличили группы и убрали отфильрованные данные, а RR не сильно выровняялся, можно сделать вывод, неравномерное удержание связано со спецификой приложения

# In[49]:


result_raw, result_grouped = get_retention(
    frame_filt, merge_data_filt, observation_date, horizon_days = 7, dimensions = ['source', 'week'], ignore_horizon= False
)
heatmap(result_grouped, name = 'Тепловая карта по источникам и неделям')


# В целом по группам из 3х недель, к 6 лайтайму можно выделить гугл, который лидирует в 2 группах из 3

# Вывод:
# Данные очень не однородны специфика приложения не дает выделить какую нибудь из групп, как лучшую по удержанию, ни по дням когорт, ни по источникам привлечения. Точно можно сказать только, что к концу 1 недели общее удержание составляет 6 процентов, к концу второй 4

# <a id="p4.2"></a>
# ### Время, проведенное в преложении пользователями

# Так как в представленных данных нет указания начала сессии и конца, сделаем своб разбивку по сессиям(предположим, что если пользователь не совершает действие более чем 15 минут, значит его сессия завершилась)
# Создадим для каждого пользователя номер сессия, а так же ее начало и конец, затем найдем время

# Данный интервал выбран исходя из понимания действий, которые совершил пользователь. Самыми долгими могут являться поиск и звонок. Что касается поиска, то считаю что 15 минут более чем достаточно в большинтсве случаев, чтобы открыть хоть какое объявление или понять, что по данному запросу нет того, что тебе нужно и начать поиск заново. Так же пользователь может относительно долго договариваться о встрече, но опять же считаю, что 15 минут для большинства пользователей вполне должно было хватить(думаю что хватило бы и 10 минут, но 15 делает определенный запас)

# In[50]:


x = '15Min'
df = merge_data.copy()
df = df.sort_values(['user_id', 'event_time'])
fd = (df.groupby('user_id')['event_time'].diff() > pd.Timedelta(x)).cumsum()
df['session_id'] = df.groupby(['user_id', fd], sort=False).ngroup() + 1
df['event_time_2'] = df['event_time']
sessions = df.groupby(['user_id','session_id']).agg({'event_time': 'first', 'event_time_2': 'last'}).rename(columns={'event_time': 'start_ses', 'event_time_2': 'end_ses'}).reset_index()
sessions['time_ses'] = (sessions['end_ses'] - sessions['start_ses']).dt.seconds
sessions


# Посмотрим количесвто сессий по часам, а так же их среднюю продолжительность

# In[51]:


sessions['hour'] = sessions['start_ses'].dt.hour
hours = sessions.groupby('hour').agg({'session_id':'count', 'time_ses':'mean'}).rename(columns={'session_id': 'count_ses', 'time_ses': 'mean_ses_time'}).reset_index()
hours


# In[52]:


fig = px.histogram(hours, x= 'hour', y='count_ses',  nbins = 24, labels = {'sum of':'количество', 'count_ses':''}).update_xaxes(categoryorder="total descending")
fig.for_each_trace(lambda t: t.update(hovertemplate=t.hovertemplate.replace('sum of','количество сессий')))
fig.for_each_yaxis(lambda a: a.update(title_text=a.title.text.replace('sum of','количесво сессий')))
fig.update_layout(
            title = {
         'text': 'количество сессий начатых в данный час',
         'y':0.95, 
         'x':0.5,
         'xanchor': 'center',
         'yanchor': 'top'
        })
fig


# Наибольшее количество сессий зафиксировано с 12 до 15 и с 20 до 21, наименьшее 5 утра

# In[53]:


fig = px.histogram(hours, x= 'hour', y='mean_ses_time',  nbins = 24, labels = {'sum of':'средняя продолжительность сессии в секундах', 'mean_ses_time':''}).update_xaxes(categoryorder="total descending")
fig.for_each_trace(lambda t: t.update(hovertemplate=t.hovertemplate.replace('sum of','средняя продолжительность сессии')))
fig.for_each_yaxis(lambda a: a.update(title_text=a.title.text.replace('sum of','средняя продолжительность сессии')))
fig.update_layout(
            title = {
         'text': 'средняя продолжительность сессии по часам ее старта',
         'y':0.95, 
         'x':0.5,
         'xanchor': 'center',
         'yanchor': 'top'
        })
fig


# Наибольшую среднюю продолжительность имеют сессии начатые 1,3 часа ночи и в 7 вечера, наименьшую в 4 утра

# In[54]:


plt.figure(figsize=(12,8))
sns.boxplot(data= sessions, y = 'time_ses',color='#AEC7E8', 
                showmeans=True, width=0.7)
plt.ylim(0,4000)
plt.title('Длительность сессии')
plt.xlabel('')
plt.ylabel('Время в приложении');
print('медиана ',round(sessions['time_ses'].median()))
print('среднее ',round(sessions['time_ses'].mean()))


# В среднем длительность сессии составляет 566 секунд(медиальное значение 259)

# In[55]:


sessions_user = sessions.groupby('user_id',as_index = False).agg({'session_id' :'count', 'time_ses':'sum'})


# In[56]:


plt.figure(figsize=(12,8))
sns.boxplot(data= sessions_user, y = 'time_ses',color='#AEC7E8', 
                showmeans=True, width=0.7)
plt.ylim(0,4000)
plt.title('Время в приложении')
plt.xlabel('')
plt.ylabel('Время в плижожении');
print('медиана ',round(sessions_user['time_ses'].median()))
print('среднее ',round(sessions_user['time_ses'].mean()))


# В среднем пользователь проводил в приложении за этот месяц 1522 секунды(медильное значение - 778)

# In[57]:


plt.figure(figsize=(12,8))
sns.boxplot(data= sessions_user, y = 'session_id',color='#AEC7E8', 
                showmeans=True, width=0.7)
plt.ylim(0,10)
plt.title('Количество сессий у пользователей')
plt.xlabel('')
plt.ylabel('количество сессий');
print('медиана ',round(sessions_user['session_id'].median()))
print('среднее ',round(sessions_user['session_id'].mean()))


# В среднем пользователь совершал за данный месяц по 3 сессии(медианное количество - 2)

# Вывод:
# Наибольшее количество сессий зафиксировано с 12 до 15 и с 20 до 21, наименьшее 5 утра
# 
# Наибольшую среднюю продолжительность имеют сессии начатые 1,3 часа ночи и в 7 вечера, наименьшую в 4 утра
# 
# В среднем длительность сессии составляет 566 секунд(медиальное значение 259)
# 
# В среднем пользователь проводил в приложении за этот месяц 1522 секунды(медильное значение - 778)
# 
# 

# <a id="p4.3"></a>
# ### Частота действий пользователей

# Рассмотрим частоту действий, которую в среднем совершает пользователь

# In[58]:


df = merge_data.groupby('user_id',as_index =False).agg({'event_time':'count'})


# In[59]:


plt.figure(figsize=(12,8))
sns.boxplot(data=df, y = 'event_time',color='#AEC7E8', 
                showmeans=True, width=0.7)
plt.ylim(0,40)
plt.title('Количество действий на пользователя')
plt.xlabel('')
plt.ylabel('Количество действий');


# В среднем пользователь совершает 17 действий, медиальное значение 9

# Рассмотрим количесвто событий по дням и источникам 

# In[60]:


merge_data['dt'] = merge_data['event_time'].dt.date


# In[61]:


fig = px.histogram(merge_data, x='dt', y='user_id', histfunc='count', labels = {'count':'количество событий', 'dt':'дата'})
fig.for_each_trace(lambda t: t.update(hovertemplate=t.hovertemplate.replace('count','количество событий')))
fig.for_each_yaxis(lambda a: a.update(title_text=a.title.text.replace('count','количество событий')))
fig.update_layout(
            title = {
         'text': 'Частота действий пользователей по дням',
         'y':0.95, 
         'x':0.5,
         'xanchor': 'center',
         'yanchor': 'top'
        })
fig


# Больше всего действий было совершено 23 октября, меньше всего 2 ноября

# In[62]:


fig = px.histogram(merge_data, x='source', y='user_id', histfunc='count', labels = {'count':'количество событий', 'source':'источник'}).update_xaxes(categoryorder="total descending")
fig.for_each_trace(lambda t: t.update(hovertemplate=t.hovertemplate.replace('count','количество событий')))
fig.for_each_yaxis(lambda a: a.update(title_text=a.title.text.replace('count','количество событий')))
fig.update_layout(
            title = {
         'text': 'Частота действий пользователей по источникам привлечения',
         'y':0.95, 
         'x':0.5,
         'xanchor': 'center',
         'yanchor': 'top'
        })
fig


# Больше всего действий совершали пользователи из источника yandex

# Дополним таблицу данным о врени суток действия, а так же о дне недели и посмотрим распределение по этим параметрам

# In[63]:


merge_data['hour'] = merge_data['event_time'].dt.hour


# In[64]:


fig = px.histogram(merge_data, x= 'hour', histfunc='count',  nbins = 24, labels = {'count':'количество действий', 'hour':'час'}).update_xaxes(categoryorder="total descending")
fig.for_each_trace(lambda t: t.update(hovertemplate=t.hovertemplate.replace('count','количество действий')))
fig.for_each_yaxis(lambda a: a.update(title_text=a.title.text.replace('count','количество действий')))
fig.update_layout(
            title = {
         'text': 'Частота действий пользователей по часам',
         'y':0.95, 
         'x':0.5,
         'xanchor': 'center',
         'yanchor': 'top'
        })
fig


# Самое активное время с 11 до 16, 20,21. Самое не активное с 1 ночи до 7 утра

# In[65]:


merge_data['week_name'] = merge_data['event_time'].apply(lambda x: x.weekday())


# Исходя из времени присвоим событию определенное время суток, и приведем номер недели к названию

# In[66]:


def times_of_day(x):
    
    if x >=5 and x <11:
        return 'утро'
    elif x >= 11 and x < 17:
        return 'день' 
    elif x >= 17 and x <23:
        return 'вечер'
    else:
        return 'ночь'
    


# In[67]:


def week_name(x):
    if x == 0:
        return 'понедельник'
    if x == 1:
        return 'вторник'
    if x == 2:
        return 'среда'
    if x == 3:
        return 'четверг'
    if x == 4:
        return 'пятница'
    if x == 5:
        return 'суббота'
    if x == 6:
        return 'воскресенье'


# In[68]:


merge_data['times_of_day'] = merge_data['hour'].apply(times_of_day)
merge_data['week_name'] = merge_data['week_name'].apply(week_name)


# In[69]:


fig = px.histogram(merge_data, x='week_name', histfunc='count', labels = {'count':'количество событий', 'week_name':'день недели'})
fig.for_each_trace(lambda t: t.update(hovertemplate=t.hovertemplate.replace('count','количество событий')))
fig.for_each_yaxis(lambda a: a.update(title_text=a.title.text.replace('count','количество событий')))
fig.update_layout(
            title = {
         'text': 'Частота действий пользователей по дням недели',
         'y':0.95, 
         'x':0.5,
         'xanchor': 'center',
         'yanchor': 'top'
        })
fig


# Больше всего действий пользователи совершают в понедельник, затем количество постепенно снижается до субботы, с небольшим ростом в вск

# In[70]:


fig = px.histogram(merge_data, x='times_of_day', y='user_id', histfunc='count', labels = {'count':'количество событий', 'times_of_day':'время суток'}).update_xaxes(categoryorder="total descending")
fig.for_each_trace(lambda t: t.update(hovertemplate=t.hovertemplate.replace('count','количество событий')))
fig.for_each_yaxis(lambda a: a.update(title_text=a.title.text.replace('count','количество событий')))
fig.update_layout(
            title = {
         'text': 'количество событий по времени суток',
         'y':0.95, 
         'x':0.5,
         'xanchor': 'center',
         'yanchor': 'top'
        })
fig


# Наибольшая активность пользователей днем, наименьшая ночью

# In[71]:


df = merge_data.groupby(['week_name','times_of_day'], as_index=False)['user_id'].agg('count').sort_values(['week_name','times_of_day'])
fig = px.sunburst(df,
                  path=['week_name','times_of_day'],
                  values = 'user_id',
                  labels = {'id':'принадлежность', 'labels':'наименование', 'user_id':'количество', 'parent':'принадлежность'},
                  height = 800 )
fig.update_layout(
            title = {
         'text': 'количество действий пользователей по дням недели',
         'y':0.97, 
         'x':0.5,
         'xanchor': 'center',
         'yanchor': 'top' 
        })
fig


# За исключением воскресенья и субботы больше всего пользователей заходят днем, а по  выходным наибольшее количество людей сидит в приложении вечером. Так же стоит отметить, что в ночь с субботы на вск количество пользователей ночью превышает количество пользователей утром.

# Вывод:
# 
# В среднем пользователь совершает 17 действий, медиальное значение – 9
# 
# Больше всего действий было совершено 23 октября, меньше всего 2 ноября
# 
# Больше всего действий совершали пользователи из источника yandex
# 
# Самое активное время взаимодействия с приложением с 11 до 16, 20,21. Самое не активное с 1 ночи до 7 утра
# 
# Больше всего действий пользователи совершают в понедельник, затем количество постепенно снижается до субботы, с небольшим ростом в вск
# 
# Наибольшая активность пользователей днем, наименьшая ночью
# 
# По будням больше всего пользователей заходят днем, а по  выходным наибольшее количество людей сидит в приложении вечером. Так же стоит отметить, что в ночь с субботы на вск количество пользователей ночью превышает количество пользователей утром.

# <a id="p4.4"></a>
# ### Конверсия в целевое действие(просмотр контактов)

# In[72]:


dataset.groupby('event_name',as_index = False).agg({'user_id':'count'}).sort_values('user_id')


# По представленной таблице, а так же по названию дейсвий делаем вывод, что нет обязательной последовательности действий, а значит нет необходимости строить и изучать воронку событий

# Создадим функцию для нахождения конверсии

# In[73]:


def get_conversion(
    profiles,
    purchases,
    observation_date,
    horizon_days,
    dimensions=[],
    ignore_horizon=False,
    target_action = 'contacts_show',
    week_max = 4
):

    if not ignore_horizon:
        last_suitable_acquisition_date = observation_date - timedelta(
            days=horizon_days - 1
        )
    result_raw = profiles.query('dt <= @last_suitable_acquisition_date')
    if 'week' in dimensions:
        weeks = math.ceil(horizon_days/7)
        last_week = week_max - weeks
        result_raw = result_raw.query('week <= @last_week')

    first_purchases = (
        purchases[purchases['event_name'] == target_action].sort_values(by=['user_id', 'event_time'])
        .groupby('user_id')
        .agg({'event_time': 'first'})
        .reset_index()
    )

    result_raw = result_raw.merge(
        first_purchases[['user_id', 'event_time']], on='user_id', how='left'
    )

    result_raw['lifetime'] = (
        result_raw['event_time'] - result_raw['first_ses']
    ).dt.days
    
    if len(dimensions) == 0:
        result_raw['cohort'] = 'All users'
        dimensions = dimensions + ['cohort']

    def group_by_dimensions(df, dims, horizon_days):

        result = df.pivot_table(
            index=dims, columns='lifetime', values='user_id', aggfunc='nunique'
        )

        result = result.fillna(0).cumsum(axis = 1)

        cohort_sizes = (
            df.groupby(dims)
            .agg({'user_id': 'nunique'})
            .rename(columns={'user_id': 'cohort_size'})
        )

        result = cohort_sizes.merge(result, on=dims, how='left').fillna(0)

        result = result.div(result['cohort_size'], axis=0)

        result = result[['cohort_size'] + list(range(horizon_days))]
        result['cohort_size'] = cohort_sizes
        return result

    result_grouped = group_by_dimensions(result_raw, dimensions, horizon_days)

    result_in_time = group_by_dimensions(
        result_raw, dimensions + ['dt'], horizon_days
    )

    return result_raw, result_grouped, result_in_time


# Изучим конверсию пользователей по источникам

# Дополним функции heatmap для верного отображения конверсии

# In[74]:


def heatmap(retention, conversion = False, name = 'удержание'):
    plt.figure(figsize=(15, 6))
    if not conversion:
        sns.heatmap(retention.drop(columns=['cohort_size', 0]), annot=True, fmt='.2%')
    else:
        sns.heatmap(retention.drop(columns=['cohort_size']), annot=True, fmt='.2%')
    plt.title(name)
    return plt.show()


# In[75]:


result_raw, result_grouped, result_in_time = get_conversion(
    frame,
    merge_data,
    observation_date,
    horizon_days,
    dimensions=['source'],
    ignore_horizon=False,
    target_action = 'contacts_show'
)
heatmap(result_grouped, conversion = True, name = 'Тепловая карта конверсии по источникам привлечения')


# Лучшую конверсию в просмотр контактов на 7 и 14 день показывают пользователи yandex

# Построим 2 графика, конверсии в динамике и без

# In[76]:


plt.figure(figsize=(20, 5)) 
report = result_grouped.drop(columns=['cohort_size'])
report.T.plot(

    grid=True, xticks=list(report.columns.values), ax=plt.subplot(1, 2, 1)
)
plt.title('Конверсия первых 14 дней с разбивкой по источникам')

report = (
    result_in_time[1]
    .reset_index()
    .pivot_table(index='dt', columns='source', values=1, aggfunc='mean')
    .fillna(0)
)
report.plot(
    grid=True, ax=plt.subplot(1, 2, 2)
)
plt.title('Динамика конверсии второго дня с разбивкой по источникам')

plt.show() 


# Конверсия пользователей yandex стабильно лучше, конверсия второго для у яндекса не всегда лучшая, но показывает стаблино хороший результат

# Изучим общую конверсию и конверсию в динамике, а так же конверсию по неделям

# In[77]:


result_raw, result_grouped, result_in_time = get_conversion(
    frame,
    merge_data,
    observation_date,
    horizon_days,
    dimensions=[],
    ignore_horizon=False,
    target_action = 'contacts_show'
)
heatmap(result_grouped,conversion = True, name = 'Тепловая карта конверсии пользователей')


# Конверсия пользователей к 7 дню 21 процент, к 14 22 процента

# In[78]:


heatmap(result_in_time, conversion = True, name = 'Тепловая карта конверсии по дням привлечения')


# Стоит выделить 3 дня с отличными результатами конверсии - 9,17 и 18 октября

# In[79]:


result_raw, result_grouped, result_in_time = get_conversion(
    frame,
    merge_data,
    observation_date,
    horizon_days,
    dimensions=['week'],
    ignore_horizon=False,
    target_action = 'contacts_show'
)
heatmap(result_grouped, conversion = True,  name = 'Тепловая карта конверсии по неделям привлечения')


# Рассчитаем общую конверсию

# In[80]:


round(merge_data[merge_data['event_name'] == 'contacts_show']['user_id'].nunique()/merge_data['user_id'].nunique(),3)


# Общая конверсия пользователей в contacts_show - 23 процента

# Пользователи первой недели показывают лучшую конверсию, чем второй

# Вывод:
# 
# Лучшую конверсию в просмотр контактов на 7 и 14 день показывают пользователи yandex
# 
# Конверсия пользователей yandex стабильно лучше, конверсия второго для у яндекса не всегда лучшая, но показывает стаблино хороший результат
# 
# Конверсия пользователей к 7 дню 21 процент, к 14 22 процента
# 
# Стоит выделить 3 дня с отличными результатами конверсии - 9,17 и 18 октября
# 
# Пользователи первой недели показывают лучшую конверсию, чем второй
# 
# Общая конверсия пользователей в contacts_show - 23 процента

# <a id="p4.5"></a>
# ### Вывод
# 
# 1) Ratation rite:
# 
# Данные очень не однородны специфика приложения не дает выделить какую нибудь из групп, как лучшую по удержанию, ни по дням когорт, ни по источникам привлечения. Точно можно сказать только, что к концу 1 недели общее удержание составляет 6 процентов, к концу второй 4
# 
# 
# 2) Время пользователей в приложении:
# 
# Наибольшее количество сессий зафиксировано с 12 до 15 и с 20 до 21, наименьшее 5 утра
# 
# Наибольшую среднюю продолжительность имеют сессии начатые 1,3 часа ночи и в 7 вечера, наименьшую в 4 утра
# 
# В среднем длительность сессии составляет 566 секунд(медиальное значение 259)
# 
# В среднем пользователь проводил в приложении за этот месяц 1522 секунды(медильное значение - 778)
# 
# 3)Частота действий пользователей
# 
# В среднем пользователь совершает 17 действий, медиальное значение – 9
# 
# Больше всего действий было совершено 23 октября, меньше всего 2 ноября
# 
# Больше всего действий совершали пользователи из источника yandex
# 
# Самое активное время взаимодействия с приложением с 11 до 16, 20,21. Самое не активное с 1 ночи до 7 утра
# 
# Больше всего действий пользователи совершают в понедельник, затем количество постепенно снижается до субботы, с небольшим ростом в вск
# 
# Наибольшая активность пользователей днем, наименьшая ночью
# 
# По будням больше всего пользователей заходят днем, а по  выходным наибольшее количество людей сидит в приложении вечером. Так же стоит отметить, что в ночь с субботы на вск количество пользователей ночью превышает количество пользователей утром.
# 
# 4)Конверсия в целевое действие
# 
# Лучшую конверсию в просмотр контактов на 7 и 14 день показывают пользователи yandex
# 
# Конверсия пользователей yandex стабильно лучше, конверсия второго для у яндекса не всегда лучшая, но показывает стаблино хороший результат
# 
# Конверсия пользователей к 7 дню 21 процент, к 14 22 процента
# 
# Стоит выделить 3 дня с отличными результатами конверсии - 9,17 и 18 октября
# 
# Пользователи первой недели показывают лучшую конверсию, чем второй
# 
# Общая конверсия пользователей в contacts_show - 23 процента
# 
# 
# 

# <a id="5"></a>
# ## Сегментация пользователей на основе действий

# <a id="p5.1"></a>
# ### Выбор метода сегментации пользователей

# По результам исследования было решено сегментировать пользователей на основе интервала дня их активности(было выявлено значительное различие количества взаимодействий пользователей с платформой), а именно разбить пользователей на группы - Ночь, Утро, Вечер, День и Мультиинтервал)
# 
# Принцип группировки:
# 
#     Найдем количество действий совершаемое пользователей в разрезе интервала дня
#     
#     Зададим процентный порог отношения действий пользователя в это время суток к общему количеству действий
#     
#     Если по какому-либо интервалу пользователь будет проходить этот порог, то он будет отнесен к группе этого интервала, в противном 
#     случае будет отнесен к группе мультиинтервал
#  
# Представленные выше 5 групп могут изменяться(объединяться исходя из количества пользователей для более равномерного распределения)

# <a id="p5.2"></a>
# ### Выделение групп пользователей

# Зададим процентный порог

# In[81]:


persent = 65


# Сгруппируем данные по пользователям в разрезе интервалов дня, отсортируем количесвто действий по времени суток по убыванию и оставим только интервал с наибольшим количество взаимодействий. Проверим пользователей на интервал и исключим тех, кто под него не подходит(это и будет группа мультиинтервал)
# 
# Построим таблицу распредления количества пользователей по заданным нами частям дня(необходимо для проверки количества пользователей в группах)

# In[82]:


df_2 = merge_data.groupby(['user_id','times_of_day'],as_index = False).agg({'event_time':'count'}).sort_values(['user_id','event_time'], ascending = False)
df_2 = df_2.groupby('user_id').agg({'times_of_day': 'first', 'event_time': 'first'}).reset_index().sort_values('user_id', ascending = False)
a = merge_data.groupby('user_id').agg({'event_time': 'count'}).reset_index().sort_values('user_id',ascending = False)
df_2 = df_2.merge(
        a[['user_id', 'event_time']], on='user_id', how='left'
    )
df_2['per'] = round(df_2['event_time_x']/df_2['event_time_y']*100,1)
df_2 = df_2[df_2['per'] > persent]
df_2.groupby('times_of_day').agg({'user_id':'count'})


# Объединим группы утро и ночь из-за небольшого количества ппользователей и потому что они идут по интервалу друг за другом
# Создадим списки пользователей относящихся к каждой части дня 

# In[83]:


user_a = list(df_2[df_2['times_of_day']== 'вечер']['user_id'])
user_b = list(df_2[df_2['times_of_day']== 'день']['user_id'])
user_c = list(df_2[(df_2['times_of_day']== 'утро')|(df_2['times_of_day']== 'ночь')]['user_id'])


# Скопируем таблицу в новую, и распределим пользователей по группа с помощью функции, использующей списки id пользователей по группам. Пользователи не попавшие в эти списки(не прошедшие процентный порог будут отнесены к мультиинтервалу)

# In[84]:


merge_data_grop = merge_data.copy()


# In[85]:


def group(x):
    if x in user_a:
        return 'Группа вечер'
    if x in user_b:
        return 'Группа день'
    if x in user_c:
        return 'Группа утро/ночь'
    else:
        return 'Группа мультиинтервал'


# In[86]:


merge_data_grop['group'] = merge_data_grop['user_id'].apply(group)
merge_data_grop.groupby('group').agg({'user_id':'nunique'})


# Вышло 4 группы пользователей(две круные, остальные две в два раза меньше)

# In[87]:


merge_data_grop.groupby('group').agg({'event_name':'count'})


# 3 из 4 групп имееют примерно одинакое количество действий, группа утро примерно в два раза уступает по этому показателю

# <a id="p5.3"></a>
# ### Вывод

# Да данном этапе было выделено 4 группы, по временному интервалу дня использования приложения:
# 
# 1) Вечер
# 2) День
# 3) Мультиинтервал
# 4) Утро/ночь

#  

#  

# <a id="p6"></a>
# ## Ответы на вопросы заказчика в разрезе выделенных групп

# <a id="p6.1"></a>
# ### Retention rate

# Создадим функцию на основе get_proffile, добавив колонку с группами к котором относятся пользователи

# In[88]:


def get_group(sessions):
    frame = (
    sessions.sort_values(by=['user_id', 'event_time'])
    .groupby('user_id')
    .agg({'event_time': 'first', 'group': 'first'})
    .rename(columns={'event_time': 'first_ses'})
    .reset_index())
    frame['dt'] = frame['first_ses'].dt.date
    frame['month'] = frame['first_ses'].astype('datetime64[M]')
    return frame


# In[89]:


frame_1 = get_group(merge_data_grop)


# Построим тепловую карту удержания

# In[90]:


observation_date = datetime(2019, 11, 3).date()
horizon_days = 14
result_raw, result_grouped = get_retention(
    frame_1, merge_data_grop, observation_date, horizon_days, dimensions = ['group'], ignore_horizon=False
)
heatmap(result_grouped, name = 'Удержание пользователей по новым группам')


# Группа мультиинтервал значительно превосходит все другие группы

# <a id="p6.2"></a>
# ### Конверсия в целевое действие

# Изучим конверсию в целевое действие

# In[91]:


result_raw, result_grouped, result_in_time = get_conversion(
    frame_1,
    merge_data_grop,
    observation_date,
    horizon_days,
    dimensions=['group'],
    ignore_horizon=False,
    target_action = 'contacts_show'
)


# In[92]:


heatmap(result_grouped,conversion = True, name = 'Конверсия пользователей по новым группам')


# Конверсия группы мульти так же показывает лучший результат, но стоит отметить так же группу день, которая занимает 2 место и значимо опережает другие группы

# <a id="p6.3"></a>
# ### Вывод
# Группа мультиинтервал значительно превосходит все другие группы по удержанию. 
# 
# Конверсия группы мульти так же показывает лучший результат,  стоит отметить так же группу день, которая занимает 2 место и значимо опережает другие группы по этому показателю, так же она была на 2 месте и по удержанию

#   

# <a id="p.7"></a>
# ## Проверка статистических гипотез

# Напишем функцию для проверки гипотез

# In[93]:


def static(data, alpha, gr1, gr2, name, grup = ['source'], event = ['event_name']):
    result = data.groupby(grup+event).agg({'user_id':'nunique'}).sort_values(grup + ['user_id'], ascending = False).reset_index()
    source = data.groupby(grup).agg({'user_id':'nunique'}).reset_index().sort_values(grup)
    source_count = np.array(source['user_id'])
    successes = np.array([result[(result[grup[0]] == gr1)&(result[event[0]] == name)]['user_id'], result[(result[grup[0]] == gr2)&(result[event[0]] == name)]['user_id']])
    trials = np.array([source[source[grup[0]] == gr1]['user_id'],source[source[grup[0]] == gr2]['user_id']])
    p1 = successes[0]/trials[0]
    p2 = successes[1]/trials[1]
    p_combined = (successes[0] + successes[1]) / (trials[0] + trials[1])
    difference = p1 - p2
    z_value = difference / math.sqrt(p_combined * (1 - p_combined) * (1/trials[0] + 1/trials[1]))
    distr = st.norm(0, 1)
    p_value = (1 - distr.cdf(abs(z_value))) * 2
    if p_value < alpha:
        return print(f'Отвергаем нулевую гипотезу: между долями есть значимая разница\np-значение: {round(p_value[0],3)}\nконверсия в {name} ({gr1}) -  {round(p1[0],2)}\nконверсия в {name} ({gr2}) - {round(p2[0],2)}')
    else:
        return print(f'Не получилось отвергнуть нулевую гипотезу, нет оснований считать доли разными\np-значение: {round(p_value[0],3)}\nконверсия в {name} ({gr1}) -  {round(p1[0],2)}\nконверсия в {name} ({gr2}) - {round(p2[0],2)}')
    


# Так как мы провдим 5 проверок в одной выборке необходимо  сделать поправку на Бонферрони, будем делить alpha при тесте на 5(количесво тестов)

# <a id="p7.1"></a>
# ### Гипотеза 1

# H0 : Конверсии групп пользователей,  установивших приложение по ссылкам из yandex и  google, в contacts_show равны.
# 
# H1 : Конверсии групп пользователей,  установивших приложение по ссылкам из yandex и  google, в contacts_show различны.

# In[94]:


static(data = merge_data_grop, alpha = 0.05/5, gr1 = 'google', gr2 = 'yandex', name = 'contacts_show')


# Не можем отвергнуть гипотизу о равенстве конверсий групп yandex и google

# <a id="p7.2"></a>
# ### Гипотеза 2

# H0 : Конверсии групп пользователей, использующих приложение днем и  в утро/ночь, в contacts_call равны
# 
# H1 : Конверсии групп пользователей, использующих приложение днем и  в утро/ночь, в contacts_call различны

# In[95]:


static(data = merge_data_grop, alpha = 0.5/5, gr1 = 'Группа день', gr2 = 'Группа утро/ночь', name = 'contacts_call', grup = ['group'])


# Отвергаем нулевую гипотизу 2 о равенстве конверсий. Конверсия группы день в contact_call в 2.5 раза больше, чем группы утро/ночь

# <a id="p7.3"></a>
# ### Гипотеза 3

# H0 : Конверсии групп пользователей, мультиинтервал с каждой из других групп, в tips_click равны
# 
# H1 : Конверсии групп пользователей, мультиинтервал с каждой из других групп, в tips_click равны

#  Данная гипотеза состоит из 3 нулевых и альтернативных подгипотез, поэтому что опровергнуть остальную, нужно чтобы все три были ошибочными

# H0 : Конверсии групп пользователей, мультиинтервал и утро/ночь, в tips_click равны
# 
# H1 : Конверсии групп пользователей, мультиинтервал и утро/ночь, в tips_click различны

# In[96]:


static(data = merge_data_grop,  alpha = 0.05/5 ,gr1 = 'Группа мультиинтервал', gr2 = 'Группа утро/ночь', name = 'tips_click', grup = ['group'])


# Отвергаем нулевую гипотизу, группа мультиинтервал имеент лучшею в 1.8 раза конверсию в tips_click чем группа утро/ночь

# H0 : Конверсии групп пользователей, мультиинтервал и день, в tips_click равны
# 
# H1 : Конверсии групп пользователей, мультиинтервал и день, в tips_click различны

# In[97]:


static(data = merge_data_grop,  alpha = 0.05/5 ,gr1 = 'Группа мультиинтервал', gr2 = 'Группа день', name = 'tips_click', grup = ['group'])


# Отвергаем нулевую гипотизу, группа мультиинтервал имеент лучшею в 1.6 раза конверсию в tips_click чем группа день

# H0 : Конверсии групп пользователей, мультиинтервал и вечер, в tips_click равны
# 
# H1 : Конверсии групп пользователей, мультиинтервал и вечер, в tips_click различны

# In[98]:


static(data = merge_data_grop,  alpha = 0.05/5 ,gr1 = 'Группа мультиинтервал', gr2 = 'Группа вечер', name = 'tips_click', grup = ['group'])


# Отвергаем нулевую гипотизу, группа мультиинтервал имеент лучшуюв 1.6 раза конверсию в tips_click чем группа вечер

# Вернемся к основной гипотизе

# H0 : Конверсии групп пользователей, мультиинтервал с каждой из других групп, в tips_click равны
# 
# H1 : Конверсии групп пользователей, мультиинтервал с каждой из других групп, в tips_click равны

# Отвергаем нулевую гипотезу, конверсия группы мультиинтервал в  tips_click больше каждой из групп в более чем 1.6 раза

# <a id="p7.4"></a>
# ### Вывод
# 
# 1) Не удалось отвергнуть гипотизу о равенстве конверсий в contact_show групп yandex и google
# 
# 2) Отвергаем гипотизу о равенстве конверсий в contacts_call. Конверсия группы день в contact_call в 2.5 раза больше, чем группы утро/ночь
# 
# 3) Отвергаем нулевую гипотезу о равентсве конверсий. Конверсия группы мультиинтервал в  tips_click больше конверсии каждой из групп более чем в 1.6 раза

#  

#  

# <a id="p8"></a>
# ## Вывод

# <a id="p8.1"></a>
# ### Общий вывод

# 1) В таблице soures 4293 строки, столько же уникальных Id пользователей и 3 типа источников привлечения пользователей, из которых yandex является самым популярным - 1934 пользователя.
# 
# 2) В таблице dataset 74197 строк, 16 уникальных событий, самое популярное из которых -  tips_show(40055), 4293 уникальных Id пользователей, наибольшее количество среди действий одного пользователя – 478
# 
# 3) Ratation rite:
# 
#     Данные очень не однородны специфика приложения не дает выделить какую нибудь из групп, как лучшую по удержанию, ни по дням 
#     когорт, ни по источникам привлечения. Точно можно сказать только, что к концу 1 недели общее удержание составляет 6 процентов, 
#     к концу второй 4
# 
# 
# 4) Время пользователей в приложении:
# 
#     4.1)Наибольшее количество сессий зафиксировано с 12 до 15 и с 20 до 21, наименьшее 5 утра
# 
#     4.2)Наибольшую среднюю продолжительность имеют сессии начатые 1,3 часа ночи и в 7 вечера, наименьшую в 4 утра
# 
#     4.3)В среднем длительность сессии составляет 566 секунд(медиальное значение 259)
# 
#     4.4)В среднем пользователь проводил в приложении за этот месяц 1522 секунды(медильное значение - 778)
# 
# 5) Частота действий пользователей
# 
#     5.1)В среднем пользователь совершает 17 действий, медиальное значение – 9
# 
#     5.2)Больше всего действий было совершено 23 октября, меньше всего 2 ноября
# 
#     5.3)Больше всего действий совершали пользователи из источника yandex
# 
#     5.4)Самое активное время взаимодействия с приложением с 11 до 16, 20,21. Самое не активное с 1 ночи до 7 утра
# 
#     5.5)Больше всего действий пользователи совершают в понедельник, затем количество постепенно снижается до субботы, с небольшим 
#     ростом в вск
# 
#     5.6)Наибольшая активность пользователей днем, наименьшая ночью
# 
#     5.7)По будням больше всего пользователей заходят днем, а по  выходным наибольшее количество людей сидит в приложении вечером. Так же стоит отметить, что в ночь с субботы на вск количество пользователей ночью превышает количество пользователей утром.
# 
# 6) Конверсия в целевое действие
# 
#     6.1)Лучшую конверсию в просмотр контактов на 7 и 14 день показывают пользователи yandex
# 
#     6.2)Конверсия пользователей yandex стабильно лучше, конверсия второго для у яндекса не всегда лучшая, но показывает стаблино 
#     хороший результат
# 
#     6.3)Конверсия пользователей к 7 дню 21 процент, к 14 22 процента
# 
#     6.4)Стоит выделить 3 дня с отличными результатами конверсии - 9,17 и 18 октября
# 
#     6.5)Пользователи первой недели показывают лучшую конверсию, чем второй
# 
# 
# 
# 
# 7) Были выделены 4 группы, по временному интервалу дня использования приложения:
# 
#      Вечер
#      День
#      Мультиинтервал
#      Утро/ночь
# 
# 
# 
# 
# 6) Группа мультиинтервал значительно превосходит все другие группы по удержанию. Конверсия группы мульти так же показывает лучший результат,  стоит отметить так же группу день, которая занимает 2 место и значимо опережает другие группы по этому показателю, так же она была на 2 месте и по удержанию
# 
# 
# 7) Гипотезы:
# 
#     7.1) Не удалось отвергнуть гипотизу о равенстве конверсий в contact_show групп yandex и google
# 
#     7.2) Отвергаем гипотизу о равенстве конверсий в contacts_call. Конверсия группы день в contact_call в 2.5 раза больше, 
#     чем группы утро/ночь
# 
#     7.3) Отвергаем нулевую гипотезу о равентсве конверсий. Конверсия группы мультиинтервал в  tips_click больше конверсии каждой из групп более чем в 1.6 раза

# <a id="p8.2"></a>
# ### Рекомендации

# Обратить основное внимание на группу мультиинтервал, она имеет значино лучшую конверсию в tips_click, чем все остальные группы. Как предложение : можно увеличить для данной когорты поток предложенных объявлений с целью увеличить общее количиство откликов по ним.
