# Metodologia Experimental

## Embeddings multimodais de imagens Sentinel-2 e texto

### 1. Objetivo

Este projeto investiga a construção de embeddings multimodais a partir de imagens de satélite Sentinel-2 e descrições textuais associadas aos mesmos patches do BigEarthNet v2.

A metodologia foi estruturada para comparar duas estratégias de representação:

1. **Representações tradicionais**, construídas diretamente a partir dos dados;
2. **Representações obtidas por modelos de Deep Learning pré-treinados**, especializados em representação visual de sensoriamento remoto e representação textual.

A pergunta experimental central é:

> Como uma representação multimodal construída a partir de métodos tradicionais se compara a uma representação multimodal construída a partir de modelos Deep Learning?

Para responder a essa questão, são definidos dois experimentos:

* **N1 — abordagem tradicional:** imagem representada por pixels + PCA e texto representado por TF-IDF + TruncatedSVD;
* **N2 — abordagem baseada em modelos Transfomer:** imagem representada por SatMAE-PP + PCA e texto representado por BGE-M3 + PCA.

O projeto separa as etapas de:

1. preparação das imagens e textos;
2. construção dos embeddings individuais;
3. redução da dimensionalidade;
4. organização dos embeddings por modalidade;
5. integração multimodal;
6. avaliação das representações obtidas.

---

# 2. Dados

Será utilizada uma amostra de aproximadamente 30.000 patches previamente selecionados do BigEarthNet v2.

Cada patch possui uma imagem multiespectral Sentinel-2 e uma descrição textual associada.

A configuração visual utilizada considera dez bandas Sentinel-2:

```text
B02
B03
B04
B05
B06
B07
B08
B8A
B11
B12
```

As bandas:

```text
B01
B09
B10
```

não fazem parte da configuração utilizada.

Os patches são organizados de forma que o `patch_id` funcione como identificador comum entre as modalidades.

Assim, uma amostra pode ser representada conceitualmente como:

```text
patch_id
   ├── imagem Sentinel-2
   └── descrição textual
```

O `patch_id` é preservado durante todo o processamento para permitir a associação posterior entre os embeddings de imagem e texto.

---

# 3. Organização geral dos experimentos

A metodologia possui **dois experimentos principais**, cada um composto por um pipeline de imagem e um pipeline de texto.

```text
                         BigEarthNet
                              │
                  ┌───────────┴───────────┐
                  │                       │
                 N1                      N2
            Tradicional              Transformer
                  │                       │
          ┌───────┴───────┐       ┌───────┴───────┐
          │               │       │               │
       IMAGEM           TEXTO   IMAGEM           TEXTO
          │               │       │               │
    Pixels + PCA     TF-IDF +   SatMAE-PP       BGE-M3
                       SVD          + PCA          + PCA
          │               │       │               │
         128D            128D    128D            128D
          │               │       │               │
          └───────┬───────┘       └───────┬───────┘
                  │                       │
                  ▼                       ▼
            Experimento N1          Experimento N2
```

Dessa forma, são produzidos quatro tipos de embedding:

```text
Not_CNN / Imagem
Not_CNN / Texto

DL / Imagem
DL / Texto
```

Esses quatro tipos de embedding compõem apenas **dois experimentos**:

```text
N1 = representação tradicional nas duas modalidades

N2 = representação baseada em modelos Transformer
     nas duas modalidades
```

Todos os embeddings finais possuem 128 dimensões, permitindo uma organização dimensional comum para as análises posteriores.

---

# 4. Organização dos dados de imagem

A configuração visual utiliza dez bandas Sentinel-2:

```text
B02, B03, B04, B05, B06,
B07, B08, B8A, B11, B12
```

**Todos os pipelines de imagem utilizam a mesma representação espacial de 96×96 pixels.**

Dessa forma, independentemente de o patch ser processado pelo pipeline tradicional ou pelo pipeline baseado em Deep Learning, sua entrada visual é representada como:

```text
[10, 96, 96]
```

Isso significa que os dois pipelines partem exatamente da mesma informação espectral e espacial.

A diferença entre as abordagens ocorre posteriormente, na forma como essa informação é transformada em uma representação vetorial.

A comparação pode ser resumida como:

```text
                  Imagem Sentinel-2
                         │
                  10 bandas
                         │
                      96×96
                         │
              ┌──────────┴──────────┐
              │                     │
             N1                    N2
              │                     │
        Pixels + PCA           SatMAE-PP
```

Essa padronização é importante para evitar que diferenças de resolução espacial sejam confundidas com diferenças decorrentes do método de representação.

---

# 5. Embedding de imagem — N1

No experimento N1, os valores dos pixels das dez bandas são utilizados diretamente para construir a representação visual.

O processo é:

```text
Imagem Sentinel-2
        ↓
Seleção das 10 bandas
        ↓
96 × 96 pixels
        ↓
Empilhamento dos canais
        ↓
Flatten
        ↓
92.160 características
        ↓
PCA
        ↓
128 dimensões
```

Como são utilizadas dez bandas e cada banda possui 96×96 pixels:

```text
10 × 96 × 96 = 92.160
```

Assim, cada patch é inicialmente representado por um vetor de **92.160 características** antes da redução de dimensionalidade.

A PCA transforma esse vetor de alta dimensionalidade em uma representação compacta de 128 dimensões.

Portanto:

```text
10 × 96 × 96
       ↓
   92.160D
       ↓
      PCA
       ↓
      128D
```

Esse pipeline não utiliza uma rede neural para aprender uma representação visual.

---

# 6. Embedding de imagem — N2

No experimento N2, o mesmo patch utilizado no experimento N1, contendo as dez bandas e resolução espacial de 96×96 pixels, é utilizado como entrada do SatMAE-PP.

O processo é:

```text
Imagem Sentinel-2
        ↓
10 bandas
        ↓
96 × 96 pixels
        ↓
SatMAE-PP
        ↓
pooler_output
        ↓
1024 dimensões
        ↓
IncrementalPCA
        ↓
128 dimensões
```

A diferença fundamental em relação ao N1 é que os pixels não são utilizados diretamente como representação final.

Eles são processados por um modelo visual pré-treinado, que transforma a informação espectral e espacial em uma representação vetorial de maior nível.

---

# 7. Embedding de texto — N1

No experimento N1, a representação tradicional dos textos utiliza TF-IDF seguida de TruncatedSVD.

O processo é:

```text
Descrição textual
        ↓
TF-IDF
        ↓
Matriz documento-termo
        ↓
TruncatedSVD
        ↓
128 dimensões
```

O TF-IDF representa os textos de acordo com a frequência dos termos e sua importância relativa no conjunto de documentos.

Como essa representação pode possuir uma quantidade elevada de características e é naturalmente esparsa, a TruncatedSVD é utilizada para obter uma representação de menor dimensionalidade.

O resultado final é:

```text
Texto
  ↓
TF-IDF
  ↓
TruncatedSVD
  ↓
128D
```

Essa abordagem não utiliza conhecimento semântico proveniente de modelos pré-treinados.

A representação é construída a partir das estatísticas observadas no corpus utilizado.

---

# 8. Pipeline de Deep Learning — N2

O experimento N2 utiliza modelos de Deep Learning **pré-treinados** para as duas modalidades.

São utilizados:

```text
Imagem → SatMAE-PP
Texto  → BGE-M3
```

Os modelos utilizados já possuem representações aprendidas durante seus respectivos processos de pré-treinamento.

O processo geral é:

```text
Imagem Sentinel-2
        ↓
SatMAE-PP
        ↓
Embedding visual 1024D
        ↓
IncrementalPCA
        ↓
128D
```

e:

```text
Descrição textual
        ↓
BGE-M3
        ↓
Embedding textual 1024D
        ↓
PCA
        ↓
128D
```

Os modelos pré-treinados são utilizados como **extratores de características**, sem atualização de seus pesos durante a geração dos embeddings.

---

# 9. Embedding visual com SatMAE-PP — N2

## 9.1 Modelo utilizado

Para as imagens Sentinel-2 foi utilizado o modelo pré-treinado:

```text
SatMAE-PP
ViT-Large
Patch8
FMoW-Sentinel Pretrain
```

O SatMAE-PP utiliza uma arquitetura baseada em Vision Transformer para aprender representações de imagens de sensoriamento remoto.

O modelo foi pré-treinado considerando dados de sensoriamento remoto, permitindo a extração de características visuais que não dependem exclusivamente das propriedades estatísticas da amostra utilizada no presente projeto.

A utilização desse modelo permite extrair características visuais de alto nível sem realizar um novo treinamento do backbone sobre os aproximadamente 30.000 patches utilizados no projeto.

---

# 10. Preparação das imagens para o SatMAE-PP

Cada patch é preparado de acordo com a entrada utilizada pelo modelo.

A representação contém:

```text
10 bandas
```

organizadas espacialmente em:

```text
10 × 96 × 96
```

As bandas são organizadas no formato esperado pelo processamento do modelo e convertidas para `float32`.

A normalização necessária é realizada pelo próprio processor associado ao modelo.

Não é aplicada uma normalização manual simplificada baseada em:

```text
pixel / 255
```

ou:

```text
pixel / 10.000
```

antes do processamento.

O objetivo é manter a preparação dos dados compatível com o procedimento esperado pelo modelo pré-treinado.

---

# 11. Extração dos embeddings visuais

O SatMAE-PP é utilizado como extrator de características.

Durante essa etapa:

* o modelo é colocado em modo de avaliação;
* seus pesos não são atualizados;
* não é realizado treinamento sobre os patches;
* o processamento é realizado em batches;
* a GPU é utilizada quando disponível;
* a inferência é realizada sem cálculo de gradientes.

Para cada patch, o modelo produz uma representação vetorial de:

```text
1024 dimensões
```

A representação utilizada é o:

```text
pooler_output
```

produzido pelo modelo.

O processo é:

```text
Imagem Sentinel-2
        ↓
10 bandas
        ↓
96 × 96
        ↓
SatMAE-PP
        ↓
pooler_output
        ↓
1024D
```

Esse vetor representa as características aprendidas pelo modelo durante seu pré-treinamento.

---

# 12. Redução dos embeddings visuais

Embora o SatMAE-PP produza embeddings de 1024 dimensões, o experimento N2 utiliza uma representação final de 128 dimensões.

Para isso, é aplicada redução de dimensionalidade por IncrementalPCA.

O processo é:

```text
SatMAE-PP
    ↓
1024D
    ↓
IncrementalPCA
    ↓
128D
```

A redução é realizada **após** a extração dos embeddings pelo SatMAE-PP.

O processo de ajuste segue a separação entre os conjuntos:

```text
TRAIN
  ↓
ajuste do IncrementalPCA

VALIDATION
  ↓
transformação

TEST
  ↓
transformação
```

Assim, os dados de validação e teste não participam do ajuste dos componentes principais.

O embedding visual final do experimento N2 possui:

```text
128 dimensões
```

---

# 13. Embedding textual com BGE-M3 — N2

## 13.1 Modelo utilizado

Para representar as descrições textuais no experimento N2 foi utilizado o:

```text
BGE-M3
```

O BGE-M3 é um modelo de representação textual baseado em Transformer pré-treinado para produzir representações semânticas de textos.

Diferentemente do TF-IDF, sua representação não depende apenas da frequência dos termos no corpus utilizado.

O modelo processa os tokens considerando seu contexto durante a geração da representação.

Assim, a representação textual obtida pelo modelo incorpora relações semânticas e contextuais aprendidas durante o pré-treinamento.

---

# 14. Extração dos embeddings textuais

As descrições textuais são processadas pelo tokenizer do BGE-M3 e posteriormente pelo modelo.

O fluxo é:

```text
Descrição textual
        ↓
Tokenização
        ↓
BGE-M3
        ↓
Representação contextual
        ↓
Embedding textual
```

A representação utilizada no pipeline possui:

```text
1024 dimensões
```

Assim:

```text
Texto
  ↓
BGE-M3
  ↓
1024D
```

O BGE-M3 é utilizado como extrator de características, sem treinamento adicional sobre o corpus do projeto durante essa etapa.

---

# 15. Redução dos embeddings textuais

Para manter uma dimensionalidade final comum entre os diferentes embeddings, os vetores produzidos pelo BGE-M3 são reduzidos para 128 dimensões.

O processo é:

```text
BGE-M3
   ↓
1024D
   ↓
PCA
   ↓
128D
```

O PCA é ajustado utilizando somente os embeddings do conjunto de treinamento.

Posteriormente, a transformação aprendida é aplicada aos conjuntos de validação e teste.

O resultado final é:

```text
Embedding textual BGE-M3 → 128D
```

---

# 16. Comparação entre os experimentos

Os dois experimentos podem ser organizados da seguinte forma:

| Experimento | Imagem          | Texto        | Característica                               |
| ----------- | --------------- | ------------ | -------------------------------------------- |
| **N1**      | Pixels + PCA    | TF-IDF + SVD | Representação tradicional                    |
| **N2**      | SatMAE-PP + PCA | BGE-M3 + PCA | Representação baseada em modelos Transformer |

Ambos os experimentos produzem embeddings finais de 128 dimensões para imagem e texto.

No caso das imagens, os dois experimentos partem exatamente da mesma entrada:

```text
10 bandas × 96 × 96
```

A diferença está no método utilizado para transformar essa entrada em uma representação vetorial.

No caso dos textos, a diferença está entre uma representação baseada nas estatísticas do corpus e uma representação produzida por um modelo Transformer pré-treinado.

---

# 17. Organização dos embeddings

Os embeddings são armazenados separadamente por modalidade e abordagem.

A estrutura utilizada é:

```text
Embeddings/
├── Not_CNN/
│   ├── Imagem/
│   │   ├── train.parquet
│   │   ├── validation.parquet
│   │   └── test.parquet
│   │
│   └── Texto/
│       ├── train.parquet
│       ├── validation.parquet
│       └── test.parquet
│
└── DL/
    ├── Imagem/
    │   ├── train.parquet
    │   ├── validation.parquet
    │   └── test.parquet
    │
    └── Texto/
        ├── train.parquet
        ├── validation.parquet
        └── test.parquet
```

Cada arquivo contém:

```text
patch_id
dim_000
dim_001
...
dim_127
```

O `patch_id` é preservado para garantir a correspondência entre as modalidades.

---

# 18. Correspondência entre imagem e texto

Como imagem e texto pertencem ao mesmo patch, os embeddings podem ser relacionados pelo `patch_id`.

Por exemplo:

```text
patch_id = X
```

pode possuir:

```text
Imagem:
DL/Imagem/train.parquet
        ↓
128D

Texto:
DL/Texto/train.parquet
        ↓
128D
```

Isso permite posteriormente construir pares:

```text
(imagem_X, texto_X)
```

e realizar análises multimodais.

Para o experimento N1:

```text
(imagem_Not_CNN_X, texto_Not_CNN_X)
```

Para o experimento N2:

```text
(imagem_DL_X, texto_DL_X)
```

Assim, a comparação multimodal pode ser realizada mantendo a correspondência entre imagem e texto por meio do mesmo `patch_id`.

---

# 19. Integração multimodal posterior

A etapa de extração dos embeddings é separada da etapa de integração multimodal.

Depois da geração dos embeddings individuais, serão comparadas duas configurações multimodais:

### N1 — representação tradicional

```text
Imagem → Pixels → PCA → 128D
                              │
                              ├── integração multimodal
                              │
Texto → TF-IDF → SVD → 128D ─┘
```

### N2 — representação baseada em modelos Transformer

```text
Imagem → SatMAE-PP → PCA → 128D
                              │
                              ├── integração multimodal
                              │
Texto → BGE-M3 → PCA → 128D ─┘
```

A integração multimodal será realizada posteriormente à geração dos embeddings individuais.

O objetivo é avaliar como a origem das representações influencia a construção e a qualidade da representação multimodal.

---

# 20. Separação entre extração, redução e integração

É importante distinguir três etapas conceitualmente diferentes.

## 20.1 Extração

Produção de uma representação individual para cada modalidade.

No N1:

```text
imagem → pixels
texto → TF-IDF
```

No N2:

```text
imagem → SatMAE-PP
texto → BGE-M3
```

## 20.2 Redução

Transformação das representações de alta dimensionalidade em vetores compactos de 128 dimensões.

No N1:

```text
Imagem:
92.160D → PCA → 128D

Texto:
TF-IDF → TruncatedSVD → 128D
```

No N2:

```text
Imagem:
1024D → IncrementalPCA → 128D

Texto:
1024D → PCA → 128D
```

## 20.3 Integração

Combinação ou alinhamento das representações de imagem e texto:

```text
embedding visual
       +
embedding textual
       ↓
representação multimodal
```

A redução para 128 dimensões **não significa, por si só, que os embeddings de imagem e texto estejam em um espaço semântico compartilhado**.

Eles possuem a mesma dimensionalidade, mas foram produzidos por métodos e modelos diferentes.

Um espaço compartilhado ou uma estratégia específica de fusão deverá ser construído em uma etapa posterior caso seja necessário realizar alinhamento multimodal.

---

# 21. Controle de data leakage

As transformações que aprendem parâmetros a partir dos dados devem utilizar exclusivamente o conjunto de treinamento.

Isso inclui:

```text
TF-IDF
TruncatedSVD
PCA
IncrementalPCA
```

O procedimento geral é:

```text
TRAIN
  ↓
ajuste dos parâmetros
  ↓
transformação

VALIDATION
  ↓
apenas transformação

TEST
  ↓
apenas transformação
```

Dessa forma, os conjuntos de validação e teste não participam da determinação dos parâmetros das transformações.

Para os modelos Transformer, o backbone é utilizado como extrator de características, sem novo ajuste de seus pesos durante a geração dos embeddings.

---

# 22. Características da comparação experimental

A comparação entre N1 e N2 foi estruturada para manter constantes as características fundamentais dos dados e modificar a estratégia de representação.

Para imagens, ambos os experimentos utilizam:

```text
10 bandas
96 × 96 pixels
```

Assim:

```text
N1:
pixels → 92.160D → PCA → 128D

N2:
pixels → SatMAE-PP → 1024D → PCA → 128D
```

A comparação visual ocorre, portanto, entre uma representação construída diretamente a partir dos pixels e uma representação obtida por um modelo visual pré-treinado.

Para textos:

```text
N1:
texto → TF-IDF → TruncatedSVD → 128D

N2:
texto → BGE-M3 → PCA → 128D
```

A comparação textual ocorre entre uma representação baseada nas estatísticas do corpus e uma representação semântica obtida por um modelo Transformer pré-treinado.

Essa estrutura permite comparar as duas estratégias em suas respectivas modalidades e, posteriormente, avaliar suas representações multimodais.

---

# 23. Matriz dos experimentos

A configuração experimental final é composta por apenas dois experimentos:

| Experimento | Representação de imagem    | Representação de texto | Objetivo                                                            |
| ----------- | -------------------------- | ---------------------- | ------------------------------------------------------------------- |
| **N1**      | Pixels + PCA               | TF-IDF + TruncatedSVD  | Construir o baseline tradicional multimodal                         |
| **N2**      | SatMAE-PP + IncrementalPCA | BGE-M3 + PCA           | Construir a representação multimodal baseada em modelos Transformer |

O **N1** representa a configuração totalmente tradicional.

O **N2** representa a configuração baseada em modelos Transformer nas duas modalidades.

Não são realizados experimentos intermediários que substituam apenas uma das modalidades.

Dessa forma, a comparação experimental concentra-se diretamente entre:

```text
N1
Tradicional
      VS
N2
Deep Learning
```

Essa escolha permite avaliar a diferença entre as duas estratégias completas de representação multimodal.

---

# 24. Estrutura final das representações

Embora sejam produzidos quatro tipos de embedding, eles estão organizados em apenas dois experimentos:

```text
                              PATCH
                                │
                 ┌──────────────┴──────────────┐
                 │                             │
                N1                            N2
           Tradicional                    Transformer
                 │                             │
          ┌──────┴──────┐               ┌──────┴──────┐
          │             │               │             │
       Imagem         Texto          Imagem         Texto
          │             │               │             │
       Pixels        TF-IDF        SatMAE-PP       BGE-M3
          │             │               │             │
         PCA           SVD             PCA           PCA
          │             │               │             │
        128D          128D            128D          128D
          │             │               │             │
          └──────┬──────┘               └──────┬──────┘
                 │                             │
                 ▼                             ▼
        Representação multimodal       Representação multimodal
                 N1                            N2
```

Assim:

```text
N1:
Imagem tradicional → 128D
Texto tradicional  → 128D

N2:
Imagem SatMAE-PP → 128D
Texto BGE-M3     → 128D
```

Todos os vetores permanecem associados ao mesmo `patch_id`.

---

# 25. Próxima etapa: integração multimodal

Com os embeddings individuais já construídos, a próxima etapa do projeto será trabalhar com a relação entre imagem e texto.

As duas configurações experimentais poderão ser utilizadas para:

* comparar similaridade entre modalidades;
* avaliar correspondência imagem–texto;
* construir um espaço compartilhado;
* realizar recuperação imagem → texto;
* realizar recuperação texto → imagem;
* testar estratégias de fusão multimodal;
* utilizar as representações combinadas em tarefas downstream.

A comparação seguirá a estrutura:

```text
N1
Imagem tradicional ──┐
                     ├──→ representação multimodal N1
Texto tradicional ───┘


N2
Imagem SatMAE-PP ────┐
                     ├──→ representação multimodal N2
Texto BGE-M3 ────────┘
```

Essa etapa será realizada posteriormente à geração e validação dos embeddings individuais.

---

# 26. Resumo da metodologia atual

A metodologia final possui **dois experimentos**, compostos por quatro pipelines de geração de embeddings.

## Experimento N1 — abordagem tradicional

### Imagem

```text
Sentinel-2
    ↓
10 bandas
    ↓
96 × 96
    ↓
92.160 características
    ↓
PCA
    ↓
128D
```

### Texto

```text
Descrição
    ↓
TF-IDF
    ↓
TruncatedSVD
    ↓
128D
```

Portanto:

```text
N1
├── Imagem → Pixels → PCA → 128D
└── Texto  → TF-IDF → SVD → 128D
```

---

## Experimento N2 — abordagem baseada em modelos Transformer

### Imagem

```text
Sentinel-2
    ↓
10 bandas
    ↓
96 × 96
    ↓
SatMAE-PP
    ↓
pooler_output
    ↓
1024D
    ↓
IncrementalPCA
    ↓
128D
```

### Texto

```text
Descrição
    ↓
BGE-M3
    ↓
1024D
    ↓
PCA
    ↓
128D
```

Portanto:

```text
N2
├── Imagem → SatMAE-PP → PCA → 128D
└── Texto  → BGE-M3 → PCA → 128D
```

---

## Comparação final

A metodologia pode ser resumida em:

```text
                         EXPERIMENTOS
                              │
                  ┌───────────┴───────────┐
                  │                       │
                 N1                      N2
            Tradicional              Transformer
                  │                       │
          ┌───────┴───────┐       ┌───────┴───────┐
          │               │       │               │
       IMAGEM           TEXTO   IMAGEM           TEXTO
          │               │       │               │
       Pixels          TF-IDF  SatMAE-PP        BGE-M3
          ↓               ↓       ↓               ↓
         PCA             SVD     PCA             PCA
          ↓               ↓       ↓               ↓
        128D            128D    128D            128D
          │               │       │               │
          └───────┬───────┘       └───────┬───────┘
                  │                       │
                  ▼                       ▼
           Multimodal N1            Multimodal N2
```

A principal característica da metodologia é que os modelos de Deep Learning utilizados no N2 **não são treinados novamente sobre os aproximadamente 30.000 patches**.

O SatMAE-PP e o BGE-M3 são utilizados como extratores de características a partir de pesos previamente aprendidos. Seus embeddings são posteriormente reduzidos para 128 dimensões.

No caso das imagens, os dois experimentos partem exatamente da mesma configuração:

```text
10 bandas × 96 × 96
```

A diferença está na estratégia de representação:

```text
N1:
pixels → 92.160D → PCA → 128D

N2:
pixels → SatMAE-PP → 1024D → PCA → 128D
```

Para o texto:

```text
N1:
texto → TF-IDF → TruncatedSVD → 128D

N2:
texto → BGE-M3 → 1024D → PCA → 128D
```

Dessa forma, o projeto compara diretamente duas estratégias completas de representação multimodal:

```text
N1 — métodos tradicionais
           VS
N2 — modelos Transformer
```

A dimensionalidade final é mantida em 128 dimensões para todas as modalidades e experimentos, enquanto o `patch_id` preserva a correspondência entre imagem e texto.

A etapa posterior de integração multimodal permitirá avaliar como essas duas estratégias de representação se comportam quando imagem e texto são considerados conjuntamente.
