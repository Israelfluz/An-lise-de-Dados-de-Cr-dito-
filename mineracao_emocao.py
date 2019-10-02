# ===============================================================================

        # O objetivo é classificar se a palavra ou frase é de alegria ou medo
    
# ===============================================================================
        

# Importando bibliotecas e programas para processamento de linguagem natural simbólica e estatística
import nltk


# Base da dados formada por um conjunto de palavras ou um texto
base = [('eu sou admirada por muitos','alegria'),
        ('me sinto completamente amado','alegria'),
        ('amar e maravilhoso','alegria'),
        ('estou me sentindo muito animado novamente','alegria'),
        ('eu estou muito bem hoje','alegria'),
        ('que belo dia para dirigir um carro novo','alegria'),
        ('o dia está muito bonito','alegria'),
        ('estou contente com o resultado do teste que fiz no dia de ontem','alegria'),
        ('o amor e lindo','alegria'),
        ('nossa amizade e amor vai durar para sempre', 'alegria'),
        ('estou amedrontado', 'medo'),
        ('ele esta me ameacando a dias', 'medo'),
        ('isso me deixa apavorada', 'medo'),
        ('este lugar e apavorante', 'medo'),
        ('se perdermos outro jogo seremos eliminados e isso me deixa com pavor', 'medo'),
        ('tome cuidado com o lobisomem', 'medo'),
        ('se eles descobrirem estamos encrencados', 'medo'),
        ('estou tremendo de medo', 'medo'),
        ('eu tenho muito medo dele', 'medo'),
        ('estou com medo do resultado dos meus testes', 'medo')]


# ============================================================================

    # O pré-processamento que vem logo abaixo com extração de palavra é um preparatório
    # da Base de dados para que depois ela venha ser passada para o NaiveBayes

# ============================================================================
    
# Início 
# Selecionando as stopwords ou o conjunto de palavras que não tem importância
stopwords = nltk.corpus.stopwords.words('portuguese')

# Remoção de stowords ou seja palavras que não são tão importantes para o entendimento da frase
def removestopwords(texto):
    frases = []
    for (palavras, emocao) in texto:
        semstop = [p for p in palavras.split() if p not in stopwords]
        frases.append((semstop, emocao))
    return frases

print(removestopwords(base))

# completamente
# completo
# comp


# Função aplicastemer que extrai o radical das palavras
def aplicastemmer(texto):
    stemmer = nltk.stem.RSLPStemmer()
    frasesstemming = []
    for (palavras, emocao) in texto:
        comstemming = [str(stemmer.stem(p)) for p in palavras.split() if p not in stopwords]
        frasesstemming.append((comstemming, emocao))
    return frasesstemming

frasescomstemming = aplicastemmer(base)


# Função de busca
def buscapalavras(frases):
    todaspalavras = []
    for (palavras, emocao) in frases:
        todaspalavras.extend(palavras)
    return todaspalavras

palavras = buscapalavras(frasescomstemming)


# Função que realiza a extração da frequência em que as palavras ocorrem no texto 
def buscafrequencia(palavras):
    palavras = nltk.FreqDist(palavras)
    return palavras

frequencia = buscafrequencia(palavras)


# Função para extrair palavras únicas
def buscapalavrasunicas(frequencia):
    freq = frequencia.keys()
    return freq

palavrasunicas = buscapalavrasunicas(frequencia)


# Função extrator que no texto (base de dados) extrai um conjunto de palavras
def extratorpalavras(documento):
    doc = set(documento)
    caracteristicas = {}
    for palavra in palavrasunicas:
        caracteristicas['%s' % palavra] = (palavra in doc)
    return caracteristicas

caracteristicasfrase = extratorpalavras(['tim', 'gole', 'nov', 'am'])

basecompleta = nltk.classify.apply_features(extratorpalavras, frasescomstemming)
print(basecompleta[11])
# fim do pré-processamento


# Criando o classificador - Passando a base de dados para o NaiveBayes
classificador = nltk.NaiveBayesClassifier.train(basecompleta)

teste = 'amor'
testestemming = []
stemmer = nltk.stem.RSLPStemmer()
for (palavrastreinamento) in teste.split():
    comstem = [p for p in palavrastreinamento.split()]
    testestemming.append(str(stemmer.stem(comstem[0])))

novo = extratorpalavras(testestemming)
print(classificador.classify(novo))
