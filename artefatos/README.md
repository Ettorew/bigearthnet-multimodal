# Recuperação Semântica de Imagens de Satélite com CLIP

## Objetivo

Construir, **do zero**, um modelo estilo **CLIP** que aprenda a relacionar imagens de satélite **Sentinel-2** com descrições em texto, de forma que seja possível:

- dado um **texto**, recuperar as imagens de satélite mais relacionadas a ele;
- dada uma **imagem**, recuperar os textos/descrições mais relacionados a ela.

Ou seja, o foco do projeto é **busca semântica** entre imagem e texto (não apenas classificação).

## Motivação

Hoje, buscar informações em bases de imagens de satélite normalmente exige conhecimento técnico (bandas espectrais, índices, classificadores treinados para cada classe específica). Um modelo de recuperação semântica permite consultar essas imagens usando **linguagem natural** — por exemplo, buscar "área de floresta próxima a corpo d'água" e receber os patches mais compatíveis, sem precisar de um classificador supervisionado para cada categoria.

## Base de dados

- **BigEarthNet**: patches de imagens do satélite **Sentinel-2**, cobrindo diversos países europeus, com rótulos de cobertura do solo baseados na **CORINE Land Cover**.
- Usaremos apenas o **Sentinel-2** (sem radar/Sentinel-1) e uma **fração** da base original, para manter o projeto simples e viável computacionalmente.
- Os rótulos de cobertura do solo servem de base para gerar as **descrições em texto** pareadas com cada imagem.

## Abordagem (CLIP do zero)

1. **Encoder de imagem**: recebe o patch Sentinel-2 (múltiplas bandas) e gera um embedding.
2. **Encoder de texto**: recebe a descrição/rótulo textual e gera um embedding de mesma dimensão.
3. **Alinhamento**: os dois embeddings são normalizados e comparados por similaridade de cosseno.
4. **Treino contrastivo (InfoNCE)**: a loss aproxima embeddings de pares corretos (imagem ↔ texto correspondente) e afasta embeddings de pares incorretos dentro do mesmo batch.
5. **Recuperação**: depois de treinado, dado um texto novo, comparamos seu embedding com os embeddings de todas as imagens (ou vice-versa) e ranqueamos pela similaridade — essa é a funcionalidade final de busca semântica.

## Escopo do projeto

- Fração reduzida da base BigEarthNet (não a base completa).
- Modelo treinado do zero (sem partir de pesos do CLIP original).
- Avaliação: verificar se, dado um texto de consulta, o modelo recupera corretamente as imagens da classe correspondente (e vice-versa).
