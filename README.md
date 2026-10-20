# Pratica-2-Visao-Computacional — Classificação de Dígitos de Hidrômetros com AlexNet, VGG16 e ResNet

**Disciplina:** Visão Computacional (VC001) — CESAR School, 2026.2
**Integrantes:** Thiago Alves Macena (tam7@cesar.school) · Henrique Bouwman (hvb@cesar.school) · Alana Cavalcanti (Accc@cesar.school) · Manuel Galvão (mfgmg@cesar.school)

## Fonte dos dados e licença

Dataset: [Yandex.Toloka Water Meters](https://www.kaggle.com/datasets/tapakah68/yandextoloka-water-meters-dataset),
licença **CC BY-NC-ND 4.0**. Este repositório **não** contém imagens do dataset, recortes gerados nem checkpoints.
O uso é restrito a fins acadêmicos e não comerciais. Os arquivos `predictions_*.csv` contêm apenas identificadores, rótulos e previsões (sem imagens).

## Como reproduzir

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Opção A: colocar o archive.zip baixado do Kaggle na raiz do repositório (ou na pasta acima) — o notebook extrai em data/
# Opção B: colocar o kaggle.json em ~/.kaggle/ (chmod 600) — o notebook baixa via kagglehub
jupyter nbconvert --to notebook --execute pratica_watermeters.ipynb --output executed.ipynb --output-dir local_outputs
# ou abrir pratica_watermeters.ipynb e executar todas as células
```

- Estrutura: `src/data_pipeline.py` (aquisição, rótulos, retificação, recortes), `src/train_lib.py` (modelos, treino, avaliação), `pratica_watermeters.ipynb` (orquestra tudo e gera as saídas), `tools/clean_notebook.py` (limpa as saídas com imagens do dataset antes do commit).
- Ambiente usado: **local, Apple M3 (16 GB), GPU via MPS**, macOS, Python 3.14.7, PyTorch 2.14.1, torchvision 0.29.1.
- Tempo total de execução do notebook: ~48 min (treino: smallcnn 47 s, alexnet 140 s, vgg16_bn 2130 s, resnet18 207 s; soma das épocas de treino, sem validação).
- Split utilizado: `splits.csv` **oficial** fornecido pelo professor (não alterado; 870/186/187 imagens em treino/validação/teste).
- Aquisição: o dataset foi obtido pelo `archive.zip` do Kaggle (mesmo conteúdo da API); o notebook também implementa o download via API (`kagglehub`) quando o zip não existe.
- Observação sobre determinismo: seed 42 em `random`, `numpy`, `torch` e no `DataLoader`; o backend MPS não garante bit-a-bit a mesma execução, então pequenas diferenças entre execuções são esperadas.

## Protocolo de treino

Protocolo do roteiro, idêntico para os quatro modelos: AdamW + OneCycleLR; `CrossEntropyLoss(label_smoothing=0.1)`; 1ª época com backbone congelado
(modelos pré-treinados); *early stopping* pelo F1-macro de validação (paciência 3); `seed=42`; augmentation `RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05))` + `ColorJitter(brightness=0.3, contrast=0.3)`, sem flip.

| Modelo | Pesos | Entrada | Batch (MPS) | LR cabeça | LR backbone | Épocas máx. | Melhor época |
|---|---|---|---|---|---|---|---|
| smallcnn | do zero | 64×48 | 128 | 1e-3 | — (todo o modelo em 1e-3) | 10 | 10 |
| alexnet | ImageNet (IMAGENET1K_V1) | 128×128 | 64 | 1e-3 | 1e-4 | 8 | 8 |
| vgg16_bn | ImageNet (IMAGENET1K_V1) | 128×128 | 16 | 1e-3 | 1e-5 | 8 | 6 |
| resnet18 | ImageNet (IMAGENET1K_V1) | 128×128 | 64 | 1e-3 | 1e-4 | 8 | 7 |

Alterações / decisões de implementação em relação ao roteiro (valem para os quatro modelos):

- **Hiperparâmetros do roteiro não foram alterados.** Batches são os de MPS/CPU da tabela do roteiro.
- *Cabeça* = última camada linear (substituída por 10 classes); *backbone* = todo o resto, inclusive as FC intermediárias (pré-treinadas) de AlexNet e VGG16. Na `smallcnn` não há LR de backbone: todo o modelo treina em 1e-3 (cabeça = linear final).
- Na 1ª época (congelada) o backbone fica em `eval()` (BatchNorm não atualiza estatísticas; dropout desligado) e só a cabeça treina. O `OneCycleLR` cobre as épocas totais.
- Normalização com média/desvio do ImageNet em todos, inclusive na `smallcnn` (que parte do zero).
- Recorte 48×64 → 128×128 por interpolação bilinear no device (sem preservar proporção). A augmentation é aplicada no recorte original, antes do redimensionamento.
- O `AdaptiveAvgPool2d` do MPS não suporta ampliar a saída (AlexNet 3×3→6×6; VGG16 4×4→7×7 com entrada 128×128). Foi usado um módulo equivalente (`AdaptiveAvgPool2dAnyDevice`, média por matrizes de pooling), verificado numericamente igual ao do PyTorch na CPU.
- `s_per_epoch` = mediana do tempo de treino por época (inclui a época 1 congelada, que é mais rápida, e o carregamento de dados com 4 *workers*; exclui a validação).
- `ms_per_image` (em `results.csv`) foi medido **no MPS**, batch 1, sobre o tensor já normalizado/redimensionado (sem pré-processamento), mediana de 200 execuções após 20 de aquecimento. A versão em CPU está em `results_extra.csv`.
- `ckpt_MB` = tamanho em disco do `state_dict` do melhor checkpoint, em MB = 10^6 bytes.

### Dataset de dígitos

Rótulo de 8 dígitos (5+3) obtido do **nome do arquivo** (string exata), preenchendo zeros à esquerda na parte inteira e à direita nos decimais; visor retificado em faixa 384×64
(perspectiva se o polígono tem 4 pontos; retângulo de área mínima caso contrário); 8 recortes de 48×64. Orientação: eixo longo do polígono → giro de 90° (visores verticais); ambiguidade de 180° resolvida pela
posição do vermelho (decimais), medida pelo percentil 95 de `R−max(G,B)` na banda central da faixa. Verificações obrigatórias atendidas: **1243** imagens válidas, **1** descartada (`id_664_value_45_2676.jpg`, 4 decimais), **9944** recortes, **0** imagens em mais de um split.

## Resultados (conjunto de teste)

| model | acc_digit | f1_macro | acc_reading | params_M | s_per_epoch | ms_per_image | ckpt_MB | best_epoch |
|---|---|---|---|---|---|---|---|---|
| smallcnn | 0.8951 | 0.8807 | 0.5722 | 0.289 | 4.42 | 0.77 | 1.2 | 10 |
| alexnet | 0.9178 | 0.899 | 0.6364 | 57.045 | 18.4 | 3.285 | 228.2 | 8 |
| vgg16_bn | 0.9372 | 0.9284 | 0.7273 | 134.31 | 303.04 | 10.626 | 537.3 | 6 |
| resnet18 | 0.9592 | 0.9547 | 0.8449 | 11.182 | 27.36 | 3.182 | 44.8 | 7 |

Complementares (`results_extra.csv`; ms em CPU com 4 threads, batch 1; GMACs por imagem):

| model | ms_per_image_cpu | gmacs | épocas executadas | acc. treino (eval, sem aug.) | tempo total de treino (s) |
|---|---|---|---|---|---|
| smallcnn | 0.9534 | 0.1159 | 10 | 0.9134 | 46.5863 |
| alexnet | 4.6924 | 0.2508 | 8 | 0.9797 | 139.7834 |
| vgg16_bn | 27.7526 | 5.1309 | 8 | 0.972 | 2129.9587 |
| resnet18 | 6.5275 | 0.5922 | 8 | 0.9958 | 206.8631 |

Figuras (pasta `figures/`):

- Matrizes de confusão normalizadas: `figures/confusion_matrices.png`
- Acurácia por posição: `figures/accuracy_by_position.png`
- Curvas de treino: `figures/training_curves.png`
- Desempenho × custo: `figures/performance_vs_cost.png`

O `classification_report` de cada modelo e a galeria dos 16 erros mais confiantes estão no notebook (a galeria tem a saída limpa por conter imagens do dataset). As previsões por recorte estão em `predictions_*.csv`.

## Respostas às questões

> **A PREENCHER PELO GRUPO**, com base nos números acima e do notebook (análise autoral, conforme as regras da prática).

**1.1** …
**1.2** …
**2.1** …
**2.2** …
**3.1** …
**3.2** Quantidade de faixas corrigidas: **198** (105 visores verticais girados 90° e 138 girados 180°, com sobreposição). Tipos de recorte errado e quantidades: …
**3.3** …
**4.1** …
**4.2** …
**4.3** …
**4.4** …
**5.1** …
**5.2** …
**5.3** …
**5.4** …
**5.5** …
**5.6** …

## Limitações

> **A PREENCHER PELO GRUPO.** O que o pipeline faz de errado e por quê? Quanto do erro restante vocês atribuem aos dados (rótulo, recorte) e quanto ao modelo? O que vocês mudariam?

## Declaração de Uso de IA

Conforme a política de uso de IA da disciplina (VC001, 2026.2). Cada integrante deve preencher a própria identificação.

| Campo | Valor |
|---|---|
| Disciplina | Visão Computacional — VC001 |
| Atividade | Prática 2 — Leitura de Hidrômetros com CNNs |
| Data | ____ / ____ / ________ |
| Aluno(a) / Matrícula | *(preencher: um bloco por integrante — Thiago Alves Macena, Henrique Bouwman, Alana Cavalcanti, Manuel Galvão)* |

**Seção A — Uso de LLM / assistente de IA:** **Sim.**

- **Modelo usado:** Claude Sonnet 5.5 (`claude-sonnet-5-5`), por meio do Claude Code (extensão do VS Code).
- **Prompts utilizados (cópia literal):**
  1. > vou realizar um trabalho pratico de visão computacional, quero que voce siga as regras do trabalho e me utilize para ter o lado critico e de entendimento teorico. na pasta ja existe oque o professor passou e aqui esta o repositorio inicial do professor: https://github.com/eron93br/vc-20262-p2
     >
     > repositorio que vamos utilizar é o: https://github.com/thiago1187/Pratica-2-Visao-Computacional.git
     >
     > meu git já esta no terminal e no read me já coloque integrantes do grupo: eu: Thiago Alves Macena (tam7@cesar.school) Henrique Bouwman (hvb@cesar.school) (Alana Cavalcanti Accc@cesar.school) Manuel Galvão (mfgmg@cesar.school)
     >
     > preciso que voce organize com boas praticas o dataset anexado, e treine utilizando AlexNet, VGG16 e ResNet18. apos isso me traga os resultados para eu avaliar os desempenhos e custos computacionais. e no final eu vou analisar os erros e limitacoes (para isso já me traga o necessario para eu analisar). o pdf já esta bem explicativo
  2. > nao sei se percebeu mas coloquei as politicas de uso da IA no trabalho
  3. Perguntas de acompanhamento sobre o andamento do treino (ex.: "quanto tempo para acabar?", "falta quanto", "acabou?").
- **Resposta/trecho gerado incorporado à entrega:** `src/data_pipeline.py`, `src/train_lib.py`, `tools/clean_notebook.py`, `pratica_watermeters.ipynb` (estrutura, células de aquisição, exploração, construção do dataset, treino, avaliação, figuras e diagnóstico de erros), `requirements.txt` e as seções factuais deste README (reprodução, protocolo, tabelas de resultados).
- **Adaptações realizadas pelo grupo:** *(PREENCHER: o que o grupo revisou, executou, corrigiu e criticou — p.ex. conferência das verificações obrigatórias, auditoria visual das faixas, reexecução do notebook, escrita das respostas e da análise de limitações).* Durante o desenvolvimento a IA corrigiu dois defeitos do próprio código (cabeça apenas no otimizador da `smallcnn`, que impedia o aprendizado; e pooling adaptativo não suportado no MPS), ambos detectados por inspeção dos resultados.

**Seção B — Uso de skill (Claude Code / SKILL.md):** Não.
**Seção C — Uso de SDD:** Não.

**Declaração de responsabilidade:** Declaramos que as informações são verdadeiras e completas, que compreendemos e somos responsáveis por todo o conteúdo entregue, incluindo as partes geradas com auxílio de IA, e que estamos cientes de que a omissão do uso de IA configura conduta academicamente desonesta.
