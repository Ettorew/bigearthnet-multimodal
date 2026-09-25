# Similaridade Estrutural entre Representações e Preservação por PCA

## 1. Contexto da análise

O objetivo desta análise é investigar como diferentes métodos de geração de embeddings representam a **estrutura de similaridade entre os mesmos patches** do conjunto BigEarthNet.

Foram consideradas três representações para cada modalidade:

- **Tradicional**
- **DL**
- **DL Sem PCA**

As comparações foram realizadas separadamente para:

- **Imagem**
- **Texto**

Para cada representação, foram calculadas as similaridades de cosseno entre pares dos mesmos 3.000 patches amostrados. Em seguida, foi calculada a **correlação de Spearman** entre os vetores de similaridade produzidos por duas representações.

Portanto, a análise não verifica se dois vetores individuais são semelhantes. Ela verifica se **dois espaços de representação preservam uma estrutura relacional semelhante entre os mesmos patches**.

---

## 2. Resultados preliminares

Os resultados obtidos foram:

| Modalidade | Comparação | Spearman ρ |
|---|---|---:|
| Imagem | Tradicional × DL | **0,898** |
| Imagem | Tradicional × Sem PCA | **0,644** |
| Imagem | DL × Sem PCA | **0,766** |
| Texto | Tradicional × DL | **0,567** |
| Texto | Tradicional × Sem PCA | **0,605** |
| Texto | DL × Sem PCA | **0,752** |

Esses resultados mostram padrões diferentes entre as modalidades.

### Imagem

Na modalidade de imagem:

- Tradicional × DL: **ρ = 0,898**
- DL × Sem PCA: **ρ = 0,766**
- Tradicional × Sem PCA: **ρ = 0,644**

A maior correlação ocorre entre a representação tradicional e a representação DL.

Isso indica que, apesar de serem produzidas por métodos diferentes, essas duas representações apresentam uma ordenação bastante semelhante das relações de similaridade entre os patches.

Por exemplo, se um determinado conjunto de patches apresenta alta similaridade segundo a representação tradicional, ele tende também a apresentar alta similaridade segundo a representação DL.

Isso **não significa que os vetores dos dois espaços sejam semelhantes**. Significa que a organização relacional dos dados é semelhante.

A comparação é, portanto, melhor interpretada como uma comparação **geométrica/relacional** entre os espaços de representação.

---

## 3. Comportamento da representação textual

Na modalidade de texto, o padrão observado é:

- Tradicional × DL: **ρ = 0,567**
- Tradicional × Sem PCA: **ρ = 0,605**
- DL × Sem PCA: **ρ = 0,752**

O resultado mais interessante é a comparação:

> **BGE-M3 original × BGE-M3 após PCA: ρ = 0,752**

Esse valor é maior do que as correlações envolvendo a representação textual tradicional.

Isso sugere, preliminarmente, que a redução dimensional mantém uma estrutura relacional mais próxima daquela presente no embedding original do BGE-M3 do que daquela presente na representação tradicional.

Em outras palavras, o PCA não parece estar transformando a representação do BGE-M3 em uma representação semelhante à produzida pelo método tradicional. A representação reduzida continua apresentando uma estrutura relativamente próxima à do espaço original do BGE-M3.

Caso esse resultado seja mantido quando a análise for realizada utilizando as **1024 dimensões completas** do embedding original, será possível afirmar que a redução dimensional preservou uma parcela considerável da estrutura relacional do embedding textual.

Uma formulação adequada para o relatório seria:

> **A redução dimensional do embedding textual produzido pelo BGE-M3 apresentou elevada concordância com a estrutura relacional do embedding original, superior à concordância observada entre o BGE-M3 e a representação textual tradicional.**

---

## 4. Comportamento da representação de imagem

Nas imagens, observa-se um padrão diferente.

A correlação entre a representação tradicional e a representação DL é particularmente elevada:

> **Tradicional × DL: ρ = 0,898**

Enquanto isso:

> **Tradicional × Sem PCA: ρ = 0,644**

e:

> **DL × Sem PCA: ρ = 0,766**

A interpretação mais apropriada não é dizer que o PCA simplesmente "aproximou" as representações.

O resultado mostra que a representação reduzida mantém uma estrutura de similaridade mais próxima da representação DL original do que da representação tradicional.

Isso pode indicar que o embedding DL e sua versão reduzida compartilham uma estrutura geométrica considerável, enquanto o método tradicional apresenta uma organização relacional mais distinta.

Entretanto, essa interpretação deve ser tratada como uma hipótese até que o experimento seja repetido corretamente com todas as dimensões do embedding Sem PCA.

---

## 5. Por que a correlação entre imagem tradicional e DL é relevante?

O SatMAE-PP é um modelo de representação desenvolvido para dados de sensoriamento remoto. Seu processo de pré-treinamento procura aprender representações úteis das imagens de satélite.

O método tradicional utilizado no projeto, por outro lado, possui uma construção completamente diferente.

Ainda assim, foi observada uma correlação de aproximadamente **0,90** entre suas estruturas de similaridade.

Isso pode indicar que os dois métodos estão capturando alguma estrutura subjacente comum presente nos patches.

Entre os fatores que potencialmente podem contribuir para essa estrutura estão:

- cobertura do solo;
- vegetação;
- padrões espectrais;
- presença de água;
- áreas urbanas;
- agricultura;
- características espaciais gerais.

Entretanto, **não é possível determinar quais fatores específicos estão sendo capturados apenas a partir da correlação de Spearman**.

Para investigar isso diretamente, seria necessário relacionar as representações com labels, metadados ou características observáveis dos patches.

---

## 6. O que o PCA está fazendo?

É importante não interpretar o PCA como uma técnica que preserva diretamente a "semântica" dos embeddings.

O PCA procura as direções de maior variância no conjunto de dados.

Assim, uma transformação:

```text
1024D
  ↓
 PCA
  ↓
128D