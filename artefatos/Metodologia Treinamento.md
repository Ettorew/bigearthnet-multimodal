# Etapa 2 — Projeto, Treinamento e Avaliação da Rede Multimodal

## 1. Objetivo

Projetar, treinar e avaliar uma rede neural profunda multimodal capaz de **associar as representações de imagem e texto correspondentes ao mesmo patch do BigEarthNet**, colocando ambas as modalidades em um espaço vetorial compartilhado.

A tarefa será tratada como um problema de **alinhamento multimodal e recuperação (retrieval)**.

Dado um patch de imagem \(I_i\) e seu texto correspondente \(T_i\), o objetivo é aprender representações:

$$
f_I(I_i) = z_i^I
$$

$$
f_T(T_i) = z_i^T
$$

tais que:

$$
sim(z_i^I,z_i^T)
$$

seja alta para pares correspondentes e baixa para pares não correspondentes.

Além de imagem → texto, o modelo também deverá funcionar na direção inversa:

$$
I \rightarrow T
$$

e

$$
T \rightarrow I
$$

---

# 2. Dados de entrada

A Etapa 1 produziu diferentes tipos de embeddings para as modalidades de imagem e texto.

Serão comparadas três configurações:

### N1 — Representações sem Deep Learning

Representações obtidas pelos métodos tradicionais utilizados na Etapa 1.

* imagem: embeddings tradicionais;
* texto: embeddings tradicionais;
* dimensionalidade final: 128 dimensões.

### N2 — Representações com Deep Learning + PCA

Representações obtidas utilizando modelos de Deep Learning e posteriormente reduzidas para 128 dimensões por PCA.

* imagem: embedding produzido pelo modelo de Deep Learning → PCA;
* texto: embedding produzido pelo modelo de Deep Learning → PCA;
* dimensionalidade final: 128 dimensões.

### N3 — Representações com Deep Learning sem PCA

Representações produzidas pelos modelos de Deep Learning sem redução por PCA.

* imagem: embedding de Deep Learning de 1024 dimensões;
* texto: embedding de Deep Learning de 1024 dimensões;
* dimensionalidade final: 1024 dimensões.

A arquitetura multimodal será adaptada para receber as dimensionalidades correspondentes a cada configuração.

---

# 3. Motivação para utilizar Projection Heads

Os embeddings produzidos pelos modelos da Etapa 1 possuem representações próprias de cada modalidade.

Mesmo quando imagem e texto possuem a mesma dimensionalidade, por exemplo:

$$
I \in \mathbb{R}^{1024}
$$

$$
T \in \mathbb{R}^{1024}
$$

isso não significa que suas coordenadas pertençam a um espaço semântico compartilhado.

Os eixos das representações foram aprendidos por modelos diferentes e possuem significados diferentes.

Por isso, não será feita diretamente a comparação entre os embeddings originais.

Serão utilizados **projection heads independentes**, um para imagem e outro para texto.

---

# 4. Arquitetura dos Projection Heads

Cada modalidade possuirá uma pequena rede neural responsável por transformar sua representação original em uma representação adequada ao alinhamento multimodal.

Para os embeddings de 1024 dimensões, será utilizada a estrutura:

$$
1024
\rightarrow
256
\rightarrow
256
$$

com uma função de ativação GELU entre as transformações lineares.

De forma conceitual:

```text
Embedding da imagem
       │
       ▼
Linear
1024 → 256
       │
       ▼
     GELU
       │
       ▼
Linear
256 → 256
       │
       ▼
Embedding projetado
```

O texto possui uma estrutura equivalente, porém com **pesos próprios**:

```text
Embedding do texto
       │
       ▼
Linear
1024 → 256
       │
       ▼
     GELU
       │
       ▼
Linear
256 → 256
       │
       ▼
Embedding projetado
```

Os pesos da imagem e do texto não são compartilhados.

Isso é necessário porque as duas modalidades possuem representações de origem diferentes.

O treinamento dos projection heads aprende transformações que permitem que as duas modalidades passem a ocupar um espaço de representação comum.

---

# 5. Por que utilizar duas camadas lineares e GELU?

Uma única transformação linear poderia ser utilizada:

$$
z = Wx+b
$$

Entretanto, nesse caso toda a transformação seria linear.

A utilização de uma função não linear, como GELU, permite que a rede aprenda transformações mais complexas.

A estrutura:

$$
Linear \rightarrow GELU \rightarrow Linear
$$

funciona como uma pequena rede neural capaz de aprender uma transformação não linear dos embeddings originais.

A função GELU introduz a não linearidade necessária para que a transformação não seja equivalente a uma única operação linear.

Os projection heads não têm como objetivo substituir os modelos que produziram os embeddings originais. Eles funcionam como uma camada de adaptação entre as representações previamente aprendidas e a tarefa multimodal.

---

# 6. Dimensionalidade do espaço compartilhado

Independentemente da dimensionalidade original, os projection heads produzirão uma representação com dimensionalidade comum:

$$
z_I,z_T \in \mathbb{R}^{256}
$$

Portanto:

```text
Imagem 1024D ──┐
               ├──→ espaço compartilhado 256D
Texto  1024D ──┘
```

Para N1 e N2, que possuem embeddings originais de 128 dimensões, a arquitetura poderá utilizar:

```text
Imagem 128D ──┐
              ├──→ espaço compartilhado 128D ou 256D
Texto  128D ──┘
```

A dimensionalidade do espaço compartilhado deverá ser mantida consistente dentro de cada experimento.

---

# 7. Normalização dos embeddings

Após os projection heads, os vetores serão normalizados utilizando normalização L2:

$$
\hat{z} =
\frac{z}{\|z\|}
$$

Com isso, todos os embeddings passam a possuir norma unitária.

A similaridade entre imagem e texto poderá então ser calculada pelo produto interno:

$$
sim(I,T)=\hat{z}_I^\top\hat{z}_T
$$

Como os vetores estão normalizados:

$$
\hat{z}_I^\top\hat{z}_T
=
\cos(\theta)
$$

Assim, a comparação utilizada pela rede será a **similaridade de cosseno**.

A normalização não elimina a direção do vetor, que é justamente a informação utilizada pela similaridade de cosseno.

---

# 8. Similaridade entre imagem e texto

Para um batch contendo \(B\) pares:

$$
(I_1,T_1), (I_2,T_2), ..., (I_B,T_B)
$$

serão produzidas duas matrizes:

$$
Z_I \in \mathbb{R}^{B\times D}
$$

$$
Z_T \in \mathbb{R}^{B\times D}
$$

A matriz de similaridade será:

$$
S = Z_I Z_T^T
$$

produzindo:

$$
S \in \mathbb{R}^{B\times B}
$$

Cada posição \(S_{ij}\) representa a similaridade entre a imagem \(I_i\) e o texto \(T_j\).

A diagonal:

$$
S_{11},S_{22},...,S_{BB}
$$

corresponde aos pares corretos.

Os elementos fora da diagonal correspondem a pares incorretos.

Exemplo:

```text
             T1      T2      T3      T4
         ┌──────────────────────────────
I1       │  ✓       ✗       ✗       ✗
I2       │  ✗       ✓       ✗       ✗
I3       │  ✗       ✗       ✓       ✗
I4       │  ✗       ✗       ✗       ✓
```

O treinamento deverá aumentar as similaridades da diagonal e diminuir as similaridades fora da diagonal.

---

# 9. Contrastive Loss

A rede será treinada utilizando uma função de perda contrastiva bidirecional.

A matriz de similaridade será utilizada em duas direções:

### Imagem → Texto

Para cada imagem \(I_i\), o modelo deverá atribuir maior similaridade ao texto correspondente \(T_i\).

### Texto → Imagem

Para cada texto \(T_i\), o modelo deverá atribuir maior similaridade à imagem correspondente \(I_i\).

Dessa forma, o treinamento não privilegia apenas uma direção da recuperação.

Conceitualmente:

```text
             Imagem → Texto
                  ↕
          Contrastive Loss
                  ↕
             Texto → Imagem
```

Os demais elementos do batch funcionam como exemplos negativos.

Assim, para um batch de 256 pares, cada exemplo possui o seu par correspondente como positivo e os demais exemplos como candidatos negativos.

---

# 10. Dataset e DataLoader

Os embeddings da Etapa 1 já estarão pré-computados.

Portanto, não será necessário executar novamente os modelos de imagem ou texto durante o treinamento multimodal.

Será utilizado um dataset contendo pares:

```text
(image_embedding, text_embedding)
```

O pareamento será mantido pelo índice:

```text
image[0] ↔ text[0]
image[1] ↔ text[1]
image[2] ↔ text[2]
...
```

O conjunto de treinamento utilizará:

* `batch_size = 256`;
* `shuffle = True`;
* `drop_last = True`.

O conjunto de validação utilizará:

* `batch_size = 256`;
* `shuffle = False`.

O `shuffle=False` na validação garante que a correspondência por índice permaneça preservada para o cálculo do retrieval.

---

# 11. Otimizador

Será utilizado o algoritmo **AdamW**:

```text
AdamW(
    lr = 1e-4,
    weight_decay = 1e-4
)
```

O AdamW será responsável por atualizar os parâmetros dos projection heads a partir dos gradientes produzidos pela contrastive loss.

O modelo original responsável pela geração dos embeddings não será treinado novamente.

Assim, a otimização estará concentrada na transformação das representações já existentes.

---

# 12. Learning Rate Scheduler

Será utilizado:

```text
CosineAnnealingLR
```

com:

```text
T_max = num_epochs
```

O learning rate será reduzido progressivamente segundo uma curva cossenoidal durante o treinamento.

Isso permite começar com atualizações maiores e reduzir gradualmente a magnitude das atualizações conforme o treinamento avança.

---

# 13. Gradient Clipping

Será utilizado:

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0
)
```

O objetivo é limitar a norma dos gradientes e evitar atualizações excessivamente grandes.

---

# 14. Validação

A validação será executada ao final de cada época.

Durante a validação:

1. O modelo será colocado em modo `eval()`.
2. Os gradientes serão desativados.
3. Todos os embeddings projetados da validação serão obtidos.
4. Os embeddings de imagem e texto serão concatenados.
5. Será calculada a matriz de similaridade/retrieval.
6. Serão calculadas as métricas de desempenho.

Serão utilizadas:

* Validation Loss;
* Recall@1;
* Recall@5;
* Recall@10;
* MRR.

As métricas serão calculadas separadamente para:

$$
Image \rightarrow Text
$$

e

$$
Text \rightarrow Image
$$

e também será calculada a média das duas direções.

---

# 15. Recall@K

Recall@K mede se o elemento correto aparece entre os \(K\) primeiros resultados do ranking.

Por exemplo:

```text
Consulta: Imagem 1

Ranking:
1. Texto 7
2. Texto 3
3. Texto 1  ← correto
4. Texto 9
5. Texto 4
```

Nesse caso:

* Recall@1 = 0;
* Recall@5 = 1.

Assim:

### Recall@1

Exige que o par correto seja o primeiro resultado.

### Recall@5

Permite que o par correto esteja entre os cinco primeiros.

### Recall@10

Permite que o par correto esteja entre os dez primeiros.

Serão calculadas as três métricas para evitar que a avaliação dependa exclusivamente de uma definição muito restritiva de acerto.

---

# 16. MRR

Será utilizada também a métrica **Mean Reciprocal Rank (MRR)**.

Para cada consulta:

$$
RR = \frac{1}{rank}
$$

Exemplos:

| Posição do par correto | Reciprocal Rank |
| ---------------------: | --------------: |
|                      1 |            1.00 |
|                      2 |            0.50 |
|                      3 |            0.33 |
|                      5 |            0.20 |
|                     10 |            0.10 |

O MRR médio permite capturar informações que o Recall@1 não consegue representar.

Por exemplo, dois modelos podem possuir o mesmo Recall@1, mas um deles pode colocar os pares incorretamente classificados predominantemente em segundo lugar, enquanto o outro pode colocá-los em posições muito mais distantes.

O MRR permite diferenciar esses comportamentos.

---

# 17. Critério de seleção do melhor modelo

O melhor checkpoint será selecionado utilizando:

$$
\boxed{R@1_{médio}}
$$

onde:

$$
R@1_{médio}
=
\frac{
R@1_{I\rightarrow T}
+
R@1_{T\rightarrow I}
}{2}
$$

A escolha dessa métrica ocorre porque o objetivo principal da rede é fazer com que o par correto ocupe a primeira posição do ranking.

Entretanto, R@1 não será utilizado isoladamente para análise.

Também serão registrados:

* Validation Loss;
* R@5;
* R@10;
* MRR;
* métricas individuais de cada direção.

Não será criada uma média artificial entre Loss e Recall, pois essas métricas possuem significados diferentes.

A Loss representa o objetivo contínuo utilizado durante a otimização, enquanto Recall e MRR representam diretamente o desempenho do sistema de retrieval.

---

# 18. Early Stopping

Será utilizado `patience = 5`.

Isso significa que o treinamento não será interrompido imediatamente quando ocorrer uma piora na validação.

Exemplo:

```text
Época 1 → melhora
Época 2 → melhora
Época 3 → piora
Época 4 → melhora
Época 5 → piora
Época 6 → piora
Época 7 → melhora
```

O treinamento continua porque novas melhorias podem ocorrer depois de uma piora temporária.

O treinamento somente será interrompido após cinco épocas consecutivas sem melhoria significativa.

Será utilizado também:

```text
min_delta = 1e-4
```

para evitar considerar variações extremamente pequenas como melhorias relevantes.

---

# 19. Salvamento do melhor checkpoint

Sempre que o `R@1 médio` da validação apresentar uma melhoria superior ao `min_delta`, serão salvos:

* parâmetros do modelo;
* estado do AdamW;
* estado do scheduler;
* época;
* Validation Loss;
* R@1 médio;
* R@5 médio;
* R@10 médio;
* MRR médio.

Ao final do treinamento, será carregado o checkpoint correspondente à melhor época.

Portanto, o modelo utilizado no teste não será necessariamente o modelo da última época.

---

# 20. Teste

O conjunto de teste será utilizado somente após o treinamento e seleção do modelo.

Fluxo:

```text
TRAIN
  │
  ▼
VALIDATION
  │
  ├── seleção do melhor checkpoint
  │
  ▼
carrega melhor modelo
  │
  ▼
TEST
```

O conjunto de teste não será utilizado para:

* escolher hiperparâmetros;
* escolher a época;
* definir early stopping;
* selecionar o melhor modelo.

Isso evita vazamento de informação do teste para o processo de treinamento.

---

# 21. Métricas finais

Para cada configuração N1, N2 e N3 serão calculadas no teste:

### Image → Text

* R@1;
* R@5;
* R@10;
* MRR.

### Text → Image

* R@1;
* R@5;
* R@10;
* MRR.

### Média das duas direções

* R@1 médio;
* R@5 médio;
* R@10 médio;
* MRR médio.

A Validation Loss também será registrada para análise, mas o desempenho final será interpretado principalmente pelas métricas de retrieval.

---

# 22. Comparação entre N1, N2 e N3

O objetivo experimental será verificar como diferentes estratégias de representação influenciam a capacidade de alinhamento multimodal.

A comparação será:

```text
                    N1
             Tradicional / 128D
                    │
                    ▼
             Rede multimodal
                    │
                    ▼
              Retrieval


                    N2
          Deep Learning + PCA
                    │
                    ▼
             Rede multimodal
                    │
                    ▼
              Retrieval


                    N3
         Deep Learning / 1024D
                    │
                    ▼
             Rede multimodal
                    │
                    ▼
              Retrieval
```

Todos os modelos serão treinados e avaliados sob o mesmo protocolo.

Serão mantidos constantes, sempre que aplicável:

* arquitetura da rede;
* função de perda;
* otimizador;
* learning rate;
* weight decay;
* scheduler;
* batch size;
* número máximo de épocas;
* patience;
* min_delta;
* divisão train/validation/test;
* métricas;
* procedimento de seleção do checkpoint.

A principal variável experimental será, portanto, a representação multimodal utilizada como entrada.

---

# 23. Hipótese experimental

A hipótese geral é que representações obtidas por modelos de Deep Learning apresentem maior capacidade de alinhamento multimodal do que representações tradicionais.

Também será investigado se a redução por PCA prejudica ou preserva essa capacidade.

Dessa forma, a comparação permitirá analisar:

$$
N1 \quad vs \quad N2 \quad vs \quad N3
$$

respondendo, respectivamente:

1. Qual é o desempenho das representações tradicionais?
2. O uso de Deep Learning melhora o alinhamento?
3. A redução por PCA prejudica significativamente as representações de Deep Learning?
4. O uso dos embeddings completos de 1024 dimensões oferece vantagem sobre a versão reduzida?

---

# 24. Fluxo completo do experimento

```text
                    ETAPA 1
                       │
                       ▼
             Embeddings pré-computados
                       │
             ┌─────────┼─────────┐
             │         │         │
             ▼         ▼         ▼
            N1        N2        N3
          128D      128D      1024D
                    PCA
             │         │         │
             └─────────┼─────────┘
                       ▼
                Projection Heads
                       │
                       ▼
             Espaço compartilhado
                       │
                       ▼
                Normalização L2
                       │
                       ▼
              Similaridade de
                 cosseno
                       │
                       ▼
            Contrastive Loss
                       │
                       ▼
                    AdamW
                       │
                       ▼
              Validação por época
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
        R@1          R@5          R@10
          │            │            │
          └────────────┼────────────┘
                       ▼
                     MRR
                       │
                       ▼
                Seleção por
                 R@1 médio
                       │
                       ▼
                 Early Stopping
                       │
                       ▼
              Melhor checkpoint
                       │
                       ▼
                    TESTE
                       │
                       ▼
             Comparação N1/N2/N3
```

# 25. Resultado esperado da Etapa 2

Ao final da etapa, será obtido um modelo multimodal treinado para associar embeddings de imagem e texto correspondentes no mesmo espaço vetorial.

A avaliação permitirá determinar não apenas se os pares corretos são recuperados, mas **em que posição eles aparecem no ranking**, utilizando conjuntamente Recall@1, Recall@5, Recall@10 e MRR.

O experimento também permitirá avaliar de forma controlada o impacto de:

* representações tradicionais;
* representações de Deep Learning;
* redução dimensional por PCA;
* manutenção dos embeddings completos.

A análise final deverá comparar N1, N2 e N3 tanto em termos de desempenho absoluto quanto em termos da qualidade do ranking multimodal produzido.
