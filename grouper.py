import warnings
import os, sys
import numpy as np
import traceback
import nltk 
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from nltk.stem import RSLPStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, homogeneity_completeness_v_measure
import matplotlib.pyplot as plt
import zipfile
import requests
from sklearn.feature_extraction.text import HashingVectorizer
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from sklearn.metrics.pairwise import cosine_similarity


def agrupar_bigramas_hash():
    freq_min = int(sys.argv[1])
    usar_ica = sys.argv[2] == '--true'
    usar_tesauro = sys.argv[3] == '--true'
    n_partitions = (int(sys.argv[4]))
    for i in np.arange(1,6):
        rodar_bigramas(freq_min, usar_ica, usar_tesauro, i, n_partitions)

def rodar_bigramas(freq_min: int, usar_ica: bool, usar_tesauro: bool, rnd: int, n_partitions: int):
    lista_k = np.arange(2, 201)
    pbar = ProgressBar()
    pbar.register()
    nltk.download('stopwords')
    erro = False
    warnings.filterwarnings("ignore")
    if not os.path.exists('dados'):
        try:
            baixar_corpus()
        except:
            print('erro: o corpus deverá ser baixado manualmente no diretório dados/corpus_tratado')
            traceback.print_exc()
            erro = True
    if not os.path.isfile('tesauro_stf.csv'):
        try:
            baixar_tesauro()
        except:
            print('erro: o tesauro deverá ser baixado manualmente no mesmo diretório do arquivo grouper.py')
            traceback.print_exc()
            erro = True
    if not os.path.exists('resultados'):
        try:
            os.mkdir('resultados')
        except:
            print('erro: não foi possivel criar o diretório de resultados; verifique as permissões e tente novamente')
            traceback.print_exc()
            erro = True       
    if not erro:
        documentos_validos = ler_documentos_validos(quantidade=10000)
        X_treino, X_teste, y_treino, y_teste = criar_holdout(documentos_validos, rnd)
        stopwords = nltk.corpus.stopwords.words('portuguese')
        diretorio = "dados/corpus_tratado/"
        le = LabelEncoder()

        opc_tesauro = '__com_crit_tesauro' if usar_tesauro  else '__sem_crit_tesauro'
        opc_ica = '__com_crit_ica' if usar_ica  else '__sem_crit_ica'
        opc_stopwords = '__removeu_sw_pt'
        exp = '__minfreq_' + str(freq_min) + opc_tesauro + opc_ica + opc_stopwords + '__hashing__seed-' + str(rnd)
        dir_experimento = 'resultados/experimento_'+str(exp)

        if not os.path.exists(dir_experimento):
            os.mkdir(dir_experimento)
        
        base_treino = criar_base_treino(exp, X_treino, y_treino, diretorio, stopwords)
        vocab = extrair_vocabulario(base_treino, freq_min, stopwords, usar_ica, usar_tesauro)

        stemmer = RSLPStemmer()
        vocab = set([stemmer.stem(token) for token in vocab])
        print('palavras unicas no vocabulario: ' + str(len(vocab)))
        print('aplicando stemmer e vocabulario na base de treino')
        ddf = dd.from_pandas(base_treino, npartitions = n_partitions)
        base_treino.teores = ddf.map_partitions(
            lambda df: df.teores.apply((
                lambda x: stem_e_vocab(x, stemmer, vocab)))).compute(scheduler ='processes')
        
        features = extrair_bigramas(base_treino.teores, vocab)
        print('total de bigramas: '+str(len(features)))

        base_teste = criar_base_teste(X_teste, y_teste, diretorio)
        ddf = dd.from_pandas(base_teste, npartitions = n_partitions)
        print('aplicando stemmer e vocabulario na base de testes')
        base_teste.teores = ddf.map_partitions(
            lambda df: df.teores.apply((
                lambda x: stem_e_vocab(x, stemmer, vocab)))).compute(scheduler ='processes')     
        print('reescrevendo documentos de teste de acordo com as features')
        ddf = dd.from_pandas(base_teste, npartitions = n_partitions)   
        base_teste.teores = ddf.map_partitions(
            lambda df: df.teores.apply((
                lambda x: vocabulzarizar_bigramas_for(x, features)))).compute(scheduler ='processes')

        p_vazios = base_teste.loc[base_teste.teores == ''].shape[0]/base_teste.shape[0]
        print('percentual de documentos vazios: '+str(p_vazios))

        base_teste = base_teste.reset_index()
        docs_teste = base_teste.teores.reset_index().teores
        hashvectorizer = HashingVectorizer(n_features=10000)

        X_kmeans = hashvectorizer.transform(docs_teste)
        y_kmeans = base_teste['assunto']
        le.fit(y_kmeans)
        y_kmeans = le.transform(y_kmeans)
        lista_scores_k = computar_scores_agrupamento(X_kmeans, y_kmeans, dir_experimento, lista_k)
        #gerar_graficos_kmeans(lista_scores_k, dir_experimento, modelo)
        np.save('resultados/' + dir_experimento + '/' + 'hashing_lista_scores_k.npy', lista_scores_k)
        print('******   dados de agrupamento do modelo hashing salvos.')

        #####MATRIZES DE SIMILARIDADE##############
        print('--------- executando analyzer para experimento '+ str(exp)+' ---------')
        sim_m = calc_matriz_sim(X_kmeans, dir_experimento)
        calcular_sim_assuntos(base_teste['assunto'], sim_m, dir_experimento)
        plt.close()

def criar_base_teste(X_teste, y_teste, diretorio):
    base_teste = pd.DataFrame(X_teste)
    base_teste['id'] = base_teste.id + '.txt'
    print("recuperando teores da base de teste")
    base_teste['teores'] = [recuperar_teor(x, diretorio) for x in tqdm(base_teste.id)]
    base_teste['assunto'] = y_teste
    return(base_teste)

def criar_holdout(documentos_validos, seed):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    X = documentos_validos.id
    y = documentos_validos.Assunto
    for index in sss.split(X, y):        
            X_treino, X_teste = X[index[0]], X[index[1]]
            y_treino, y_teste = y[index[0]], y[index[1]]
    return (X_treino, X_teste, y_treino, y_teste)

def extrair_bigramas(teores, vocab):
    print('extraindo bigramas da base de treino')
    vectorizer = TfidfVectorizer(ngram_range = (2,2))
    vectorizer.fit(teores)

    bigramas = vectorizer.get_feature_names()
    print('quantidade de bigramas antes do corte: '+ str(len(bigramas)))
    print('selecionando bigramas que contém o vocabulário')
    vocab_bigrama = []
    for bigrama in bigramas:
        if all(item in vocab for item in bigrama.split(' ')):
            vocab_bigrama.append(bigrama)
    
    vectorizer = TfidfVectorizer(vocabulary = vocab_bigrama, ngram_range = (2,2))
    vectorizer.fit(teores)
    features = vectorizer.get_feature_names()
    return(features)

def calc_matriz_sim(vetores, dir_experimento):
    print("calculando matriz de similaridade entre os documentos")
    return cosine_similarity(vetores)

def calcular_sim_assuntos(assuntos, sim_docs, dir_experimento):
    print('calculando a similaridade entre assuntos')
    lista_sim_assuntos = []
    lista_assuntos = assuntos.unique()
    
    for i in tqdm(range(0, lista_assuntos.shape[0])):
        for j in range(0, lista_assuntos.shape[0]):    
            assunto_a = lista_assuntos[i]
            assunto_b = lista_assuntos[j]

            indices_a = assuntos[assuntos == assunto_a].index.values
            indices_b = assuntos[assuntos == assunto_b].index.values
            x = sim_docs[np.ix_(indices_a,indices_b)]

            #Se os assuntos forem os mesmos, apenas o triângulo superior (sem a diagonal principal) deve ser consideradono cálculo da média
            #caso contrário, todos os elementos podem ser considerados
            if assunto_a == assunto_b:
                ind_sup = np.triu_indices(max(len(indices_a), len(indices_b)), k=1)
                sim = x[ind_sup].mean()
            else:
                sim = sim_docs[np.ix_(indices_a,indices_b)].mean()
            lista_sim_assuntos.append((assunto_a, assunto_b, sim))
    lista_sim_assuntos = pd.DataFrame.from_records(lista_sim_assuntos, columns = ['assunto_a', 'assunto_b', 'sim_cos'])
    pivot = lista_sim_assuntos.pivot(index='assunto_a', columns='assunto_b', values='sim_cos')
    pivot.to_csv(dir_experimento+'/sim_assuntos_hashing.csv')
    plt.cla()

#retorna dataframe com duas colunas contendo, respectivamente, ids e assunto dos documento
def ler_documentos_validos(quantidade=40009, min_freq_assunto = 50, min_palavras_documento = 50):
    df = pd.read_csv('dados/corpus_tratado/metadados.csv')[0:quantidade]
    documentos_validos = filtrar_assuntos(min_freq_assunto, df)
    documentos_validos = filtrar_documentos_curtos(min_palavras_documento, documentos_validos)
    print('total de documentos no corpus: '+str(documentos_validos.shape[0]))
    return documentos_validos

def filtrar_assuntos(min_freq_assunto, df):
    print('filtrando assuntos com frequencia minima ' + str(min_freq_assunto))
    df_validos = pd.DataFrame(df.groupby("Assunto").size()).reset_index()
    df_validos.columns = ["assunto", "quant"]
    df_validos = df_validos.loc[df_validos.quant >= min_freq_assunto]
    documentos_validos = df[df.Assunto.isin(df_validos.assunto)][["id", "Assunto"]].reset_index().drop('index', axis = 1)
    diretorio = "dados/corpus_tratado/"
    documentos_validos['arquivo'] = documentos_validos.id + '.txt'
    documentos_validos['teores'] = [recuperar_teor(x, diretorio) for x in tqdm(documentos_validos['arquivo'])]
    return documentos_validos

def recuperar_teor(idx, diretorio):
    with open(diretorio + idx, "r", encoding='utf-8') as f:
        contents = f.read()
    return contents

def filtrar_documentos_curtos(min_palavras_documento, documentos_validos):
    print('filtrando assuntos com no minimo ' + str(min_palavras_documento) + ' palavras')
    documentos_validos['validos'] = documentos_validos.teores.apply(lambda x: 0 if len(x.split(' ')) <= min_palavras_documento else 1)
    documentos_validos = documentos_validos[documentos_validos['validos'] == 1].drop(['arquivo', 'teores', 'validos'], axis = 1).reset_index().drop('index', axis = 1)
    return documentos_validos

#recebe o número do experimento e os ids da base de treino
#retorna a base de treino enriquecida dos teores e dos assuntos
def criar_base_treino(exp, X_treino, y_treino, diretorio, stopwords):
    X_treino = pd.DataFrame(X_treino)
    X_treino['id'] = X_treino.id + '.txt'

    print("preparando documentos para extração do vocabulário:")
    X_treino['teores'] = [recuperar_teor(x, diretorio) for x in tqdm(X_treino.id)]
    X_treino['assunto'] = y_treino
    return X_treino

#recebe o id de um documento e o diretorio onde ele se encontra, como strings
#retorna o texto contido neste documento
def recuperar_teor(idx, diretorio):
    with open(diretorio + idx, "r", encoding='utf-8') as f:
        contents = f.read()
    return contents

def extrair_vocabulario(corpus, corte_freq, stopwords, usar_ica, usar_tesauro):
    if usar_ica:
        
        print("extraindo termos com base no ICA")
        termos_ica = sel_termos_ica(corpus)
    else:
        termos_ica = set()

    print("extraindo termos com base na frequência - geralmente leva menos de 4 minutos")
    termos_freq = sel_termos_freq(corpus, corte_freq, stopwords, True)

    if usar_tesauro:
        print("extraindo termos do tesauro")
        termos_tesauro = sel_termos_tesauro()
    else:
        termos_tesauro = set()
    vocabulario = termos_tesauro.union(termos_ica).union(termos_freq)
    print("***************extração de vocabulário concluída: " + str(len(vocabulario)) + ' palavras******************')
    return vocabulario

#recebe como entrada o dataframe do conjunto de treinamento contendo id, teores e assunto
#retorna conjunto com termos cujo ICA é maior que a média observada no vocabulário extraído dos teores
def sel_termos_ica(X_treino):
    #agrupa teores por assunto, concatenando-os
    df_assuntos = pd.DataFrame()
    for classe in tqdm(X_treino['assunto'].unique()):
        concat = ' '.join(X_treino.loc[X_treino['assunto'] == classe].teores)
        df_assuntos = pd.concat([df_assuntos, pd.DataFrame([(classe, concat)])], ignore_index = True)
    df_assuntos.columns = ['assuntos', 'teores']
    print("-processando strings do corpus")
    #calcula o ICA dos termos que aparecem no conjunto de treino
    stopwords = nltk.corpus.stopwords.words('portuguese')
    df_assuntos['teores'] = df_assuntos.teores.str.replace('\n', ' ').str.strip()
    print("-treinando vetorizador")
    vectorizer = TfidfVectorizer(stop_words = stopwords, max_df = 50, smooth_idf = False)
    vec = vectorizer.fit(df_assuntos.teores)
    
    #retorna conjunto com termos cujo ICA é maior que a média de ICA do vocabulário
    df = pd.DataFrame(list(zip([k for k, v in vec.vocabulary_.items()], vec.idf_)), columns = ['termo', 'idf']).sort_values(by='idf')
    estats_idf = pd.DataFrame(df.idf.describe())
    corte_idf = estats_idf.loc['mean',:]
    df = df[df.idf >= corte_idf[0]]
    print("-ICA processado")
    return(set(df.termo))

#recebe uma série do pandas contendo o corpus, o valor mínimo de frequência das palavras e um conjunto de stopwords
#retorna um conjunto de palavras que aparecem no mínimo freq_min vezes no corpus
def sel_termos_freq(corpus, freq_min, stopwords, remover_stopwords_pt):
    contagem = corpus.teores.str.split(expand=True).stack().value_counts()
    if remover_stopwords_pt:
        p_validas = set(contagem.index) - set(stopwords)
    else:
        p_validas = set(contagem.index)
    contagem = contagem.loc[contagem.index.intersection(p_validas)][contagem>=freq_min]
    return(set(contagem.index))

#retorna os termos do tstf compostos por apenas uma palavra
def sel_termos_tesauro():    
    tesauro = pd.read_csv("tesauro_stf.csv")
    tesauro['unico'] = tesauro.apply(lambda x: 1 if x['termo'].strip().count(' ') == 0
                                        else 0, axis = 1)
    termos_tesauro = tesauro.loc[tesauro.unico == 1].reset_index().termo.str.lower()
    return(set(termos_tesauro))

def computar_scores_agrupamento(X, y, dir_experimento, lista_k):
    lista_scores_k = []
    for k in tqdm(lista_k):
        kmeans = KMeans(n_clusters=k, random_state=0, verbose = 0).fit(X)
        preds_kmeans = kmeans.predict(X)
        sil_score_kmeans = silhouette_score(X, preds_kmeans)
        ari_kmeans = adjusted_rand_score(y, preds_kmeans)
        hcv_kmeans = homogeneity_completeness_v_measure(y, preds_kmeans)
        lista_scores_k.append((k, 
                              sil_score_kmeans, 
                              ari_kmeans, 
                              hcv_kmeans[0],
                              hcv_kmeans[1], 
                              hcv_kmeans[2]))
    return(lista_scores_k)

def stem_e_vocab(doc, stemmer, vocab):
    return(vocabularizar_documento(' '.join([stemmer.stem(token) for token in doc.split()]), vocab))

def vocabulzarizar_bigramas_comp(doc, lista_bigramas):
    palavras = doc.split()
    bigramas_velho = nltk.ngrams(palavras, 2)
    bigramas_novo = [' '.join(bigrama) for bigrama in bigramas_velho if ' '.join(bigrama) in lista_bigramas]
    return(' '.join(bigramas_novo))

def vocabulzarizar_bigramas_for(doc, lista_bigramas):
    palavras = doc.split()
    bigramas_velho = nltk.ngrams(palavras, 2)
    bigramas_novo = []
    for bigrama in bigramas_velho:
        candidato = ' '.join(bigrama)
        if candidato in lista_bigramas:
            bigramas_novo.append(candidato)
    return(' '.join(bigramas_novo))

def vocabularizar_documento(documento, vocab):
    tokens_velho = documento.split()
    tokens_novo = []
    for token in tokens_velho:
        if token in vocab:
            tokens_novo.append(token)
    return(' '.join(tokens_novo))

def baixar_corpus():
    file_id = '14UJLcDDZX5dIX6CJVCFkNHXZttTKfbC5'
    destination = 'corpus_tratado.zip'
    print('Baixando arquivo corpus. Por favor aguarde.')
    download_file_from_google_drive(file_id, destination, 150000000)
    with zipfile.ZipFile('corpus_tratado.zip') as zf:
        for member in tqdm(zf.infolist(), desc='Extraindo corpus: '):
            try:
                zf.extract(member, 'dados/')
            except zipfile.error:
                pass
    os.remove(destination)

def baixar_tesauro():
    file_id = '1CAKVxcX9RxX5gSq3XzzGl7-JMCOPmBop'
    destination = 'tesauro_stf.csv'
    print('Baixando tesauro. Por favor aguarde.')
    download_file_from_google_drive(file_id, destination, 150000000)

def download_file_from_google_drive(id, destination, file_size):
    URL = 'https://docs.google.com/uc?export=download'

    session = requests.Session()
    response = session.get(URL, params = { 'id' : id }, stream = True)

    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    CHUNK_SIZE = 32*1024
    # TODO: this doesn't seem to work; there's no Content-Length value in header?
    total_size = file_size

    with tqdm(desc=destination, total=total_size, unit='B', unit_scale=True) as pbar:
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    pbar.update(CHUNK_SIZE)
                    f.write(chunk)

if __name__ == "__main__":
    agrupar_bigramas_hash()
