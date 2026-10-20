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

### Parte 1

**1.1** A pasta images tem a foto original, a masks tem a máscara do visor e a collage tem a foto com o contorno desenhado, só pra visualizar. No data.csv tem o nome do arquivo, o valor da leitura em m³ e o polígono do visor. Pra treinar usamos images e location ou masks, a collage não porque já vem com desenho em cima. O notebook confirmou que as 3 pastas têm 1244 arquivos cada, todos com linha no CSV e nenhum ilegível, e que a máscara é o próprio polígono desenhado (IoU de 1,000 em 150 imagens).

**1.2** O rótulo é por imagem, a leitura inteira de 8 dígitos. Não tem rótulo por dígito. Então a gente localiza o visor, fatia em 8 e assume que a fatia i é o dígito i. Isso é um rótulo fraco, e quando o corte erra vira ruído. Por isso o dataset de dígitos herda os erros do pipeline, e parte do erro que aparece depois não é do modelo.

### Parte 2

**2.1** A fonte mais confiável é o nome do arquivo. Ele bate com o value do CSV em 1244 de 1244 linhas, mas o CSV tem ruído de ponto flutuante em 300 linhas e perde zeros no final, e o nome guarda a string exata. Máscara e polígono só dizem onde está o visor. Ressalva, ninguém conferiu o rótulo contra a imagem. Ao olhar os erros que os 4 modelos fazem juntos, vimos casos em que o rótulo parece não bater com o recorte (imagens 420 e 372), mas não dá pra separar se é rótulo errado ou fatia deslocada.

**2.2** O 0 é a classe majoritária, com 3549 dos 9944 recortes, 35,7 por cento. Isso acontece porque são 5 dígitos inteiros e a maioria dos medidores está abaixo de 1000 m³, então sobra zero à esquerda. Por isso a acurácia sozinha engana e usamos F1 macro. Nas posições 0 e 1 o 0 aparece em 99,8 e 95,9 por cento dos casos, e nas posições 3 a 7 os dígitos ficam em torno de 10 por cento cada, então o desbalanceamento vem só do começo da leitura.

### Parte 3

**3.1** Os 8 recortes da mesma foto têm o mesmo fundo, luz e medidor. Se dividir por recorte, quase o mesmo exemplo cai no treino e no teste e a métrica fica inflada. Dividimos por imagem e deu vazamento zero, com 870, 186 e 187 imagens. Todos usam o mesmo splits.csv pra comparação ser justa. Na prática o split por imagem também faz o teste simular um medidor novo, nunca visto no treino.

**3.2** Foram corrigidas 198 faixas, 105 verticais e 138 giradas 180 graus, com sobreposição. Outros problemas foram 73 faixas sem evidência de cor pra checar orientação, 113 com aspecto fora de 4 a 7 e 36 imagens com rótulo terminando em 000, que eu inferi serem visores sem decimais. Na auditoria de 60 faixas, nenhuma estava de cabeça para baixo e cerca de 23 (38 por cento) tinham o último dígito meio rolado, contagem visual, então é uma estimativa. Não contei polígono impreciso, por isso não entra na tabela.

| Tipo de problema | Como medi | Quantidade | % |
|---|---|---|---|
| Faixa corrigida (90° e/ou 180°) | cálculo no pipeline | 198 de 1243 | 15,9 |
| Sem evidência de cor para orientação | diferença de vermelho < 8 | 73 de 1243 | 5,9 |
| Aspecto do polígono fora de 4 a 7 | medido no polígono | 113 de 1243 | 9,1 |
| Rótulo terminando em 000 (visor sem decimais, inferido) | contagem nos rótulos | 36 de 1243 | 2,9 |
| Último dígito meio rolado | contagem visual em 60 faixas | cerca de 23 de 60 | cerca de 38 |
| Faixa de cabeça para baixo | auditoria visual de 60 faixas | 0 de 60 | 0 |

**3.3** Dividir em larguras iguais assume 8 tambores idênticos e bem alinhados. O aspecto médio foi 5,12 contra 6,0 esperado, então o corte pode cair no meio do dígito e o rótulo não bate com a fatia. Nas imagens sem decimais a acurácia por dígito cai pra uma faixa de 41 a 53 por cento. Dá pra corrigir detectando as fronteiras dos tambores ou usando um modelo de sequência com CTC. Eu escolheria detectar as fronteiras (perfil de projeção vertical), porque mantém o mesmo classificador e dá pra medir o ganho com o mesmo protocolo. O CTC resolve mais coisa, mas muda o modelo e o treino inteiros.

### Parte 4

**4.1** Os pesos pré treinados foram ajustados com a média e o desvio do ImageNet. Sem normalizar do mesmo jeito, a escala de entrada muda e a transferência tende a piorar. Não medimos o efeito de tirar a normalização. Na smallcnn, que parte do zero, a normalização do ImageNet não tem justificativa própria, usamos igual só pra manter o protocolo idêntico.

**4.2** Os parâmetros são smallcnn 0,29M, alexnet 57M, vgg16 134M e resnet18 11M. Na AlexNet, 96 por cento ficam nas camadas totalmente conectadas, e na VGG16 são 89 por cento. Parâmetro pesa na memória, não no tempo, porque a AlexNet tem menos GMACs que a ResNet18, 0,25 contra 0,59, mesmo com 5 vezes mais parâmetros. Então contar só parâmetros engana na hora de comparar custo: pro checkpoint importa, mas a latência segue mais os GMACs, e a VGG16 é a única cara nos dois (134M e 5,13 GMACs).

**4.3** Treinar só a cabeça é barato e estável, mas o backbone não se adapta aos dígitos. O ajuste fino completo adapta tudo, custa mais e pode estragar features úteis. Como a cabeça nova começa aleatória, uma época congelada ajuda ela a se ajustar antes. O F1 de validação saltou da época 1 pra 2, de 0,408 pra 0,802 na alexnet e de 0,113 pra 0,889 na resnet. Esse salto mostra que a cabeça sozinha não bastava, o backbone precisou se adaptar. A época 1 também é bem mais barata, 6,5 s contra uns 18 s na alexnet, porque não calcula gradiente no backbone.

**4.4** Dígito não é simétrico. Espelhar cria glifos que não existem ou troca a identidade, tipo 2 e 5, ou 6 e 9 com flip vertical. Rotação pequena, translação e brilho imitam variação real de câmera e luz. Por exemplo, um 2 espelhado fica parecido com um 5 e o rótulo continuaria 2, o que ensinaria o modelo errado.

### Parte 5

**5.1** A ResNet18 teve o melhor F1 macro, 0,9547, e o ranking é o mesmo da acurácia, resnet, vgg, alexnet, smallcnn. O F1 ficou abaixo da acurácia em todos. Minha hipótese era que as classes diferentes de 0 são um pouco piores, e conferindo o classification_report ela se sustenta só em parte: o F1 da classe 0 (0,925 a 0,968) fica acima do F1 macro em todos os modelos, mas a pior classe varia, é a 6 na smallcnn, a 7 na alexnet, a 1 na vgg16 e a 7 na resnet. O McNemar favorece a resnet sobre a vgg, 40 contra 7 discordâncias, com a ressalva de que recortes da mesma imagem não são independentes.

**5.2** Cada modelo erra de um jeito. smallcnn e vgg jogam muito dígito pro 0, tipo 6 pra 0 em 14 casos na smallcnn, e 1 pra 0 em 9 na vgg. A alexnet é diferente, o maior erro dela é 1 pra 7, 6 casos. Na resnet os erros são poucos e espalhados. Dos 16 erros compartilhados pelos 4 modelos com maior confiança (de 46 no total), 5 eram recorte deslocado, 5 rótulo inconsistente (o recorte mostra um dígito limpo diferente do rótulo), 3 meio rolado, 2 ilegíveis e só 1 erro do modelo. Ou seja, o que os 4 erram juntos vem quase todo do pipeline e dos dados, e se repete nas mesmas imagens: a 420 aparece 3 vezes, e a 372, 480, 114 e 986 aparecem 2 vezes cada. Essa classificação é visual, então tem julgamento nela.

**5.3** A leitura exige os 8 dígitos certos, então um erro só invalida tudo. Mas os erros se concentram, na resnet 158 das 187 imagens não têm erro nenhum, e a posição 7 é a pior em todos os modelos, provavelmente pelo tambor girando. Na auditoria de 60 faixas cerca de 38 por cento tinham o último dígito meio rolado, então o rótulo ali é ambíguo mesmo pra um olho humano.

**5.4** Eu usaria a ResNet18, com acurácia de leitura de 0,845, 45 MB e uns 6,5 ms por recorte na CPU. A smallcnn é minúscula, 1,2 MB, mas erra 43 por cento das leituras. Ressalva, medimos num M3 e não num celular. Eu adicionaria um limiar de confiança pra pedir uma nova foto, já que os erros têm confiança média de 0,4 a 0,5 contra 0,72 a 0,88 nos acertos, mas isso não foi testado.

**5.5** A ResNet ganha 2,2 pontos na acurácia por dígito e 11,8 na leitura, com 12 vezes menos parâmetros e treino bem mais rápido. Provavelmente por causa das conexões residuais, da BatchNorm e do pooling global, mas isso é teoria, não medimos. Também teve taxa de aprendizado diferente, 1e-4 contra 1e-5 na VGG. Como a seed foi uma só, eu trato a vantagem como forte (McNemar) mas sem atribuir tudo à arquitetura.

**5.6** Em acurácia por dígito foi quase competitiva, 0,895 contra 0,918 da alexnet com 200 vezes menos parâmetros. Em leitura completa não, 0,572 contra 0,845 da resnet, porque os erros se somam nos 8 dígitos. Ela ainda estava melhorando na época 10, então está subtreinada. Pra mim isso indica que o dígito isolado é fácil e um modelo pequeno já pega quase tudo, e o que separa os modelos é a robustez nos casos difíceis, que é o que decide a leitura inteira.

## Limitações

O pipeline divide a faixa em 8 fatias iguais e assume que a fatia i é o dígito i. Isso erra nas imagens sem decimais (36 imagens, acurácia por dígito de 41 a 53 por cento) e no último dígito, que costuma estar meio rolado. A maior parte do erro restante da ResNet18 parece vir dos dados e do recorte, não do modelo: 46 dos 61 erros dela são compartilhados pelos 4 modelos. O teste tem só 187 imagens, a seed foi uma só e a latência foi medida num M3, não num celular. Eu mudaria a divisão da faixa pra detectar as fronteiras dos tambores.

## Declaração de Uso de IA

Conforme a política de uso de IA da disciplina (VC001, 2026.2). Declaração única do grupo.

| Campo | Valor |
|---|---|
| Disciplina | Visão Computacional — VC001 |
| Atividade | Prática 2 — Leitura de Hidrômetros com CNNs |
| Data | 20 / 10 / 2026 |
| Alunos / Matrícula | Thiago Alves Macena (matrícula: ____) · Henrique Bouwman (matrícula: ____) · Manuel Galvão (matrícula: ____) · Alana Cavalcanti (matrícula: ____) |

**Seção A — Uso de LLM / assistente de IA:** **Sim.**

- **Modelo usado:** Claude Sonnet 5.5 (`claude-sonnet-5-5`), por meio do Claude Code (extensão do VS Code).
- **Prompts utilizados (cópia literal):**
  1. Prompt inicial (código, dataset, treino e resultados):
  > vou realizar um trabalho pratico de visão computacional, quero que voce siga as regras do trabalho e me utilize para ter o lado critico e de entendimento teorico. na pasta ja existe oque o professor passou e aqui esta o repositorio inicial do professor: https://github.com/eron93br/vc-20262-p2
  >
  > repositorio que vamos utilizar é o: https://github.com/thiago1187/Pratica-2-Visao-Computacional.git
  >
  > meu git já esta no terminal e no read me já coloque integrantes do grupo: eu: Thiago Alves Macena (tam7@cesar.school)
  > Henrique Bouwman (hvb@cesar.school)
  > (Alana Cavalcanti Accc@cesar.school)
  > Manuel Galvão (mfgmg@cesar.school)
  >
  > preciso que voce organize com boas praticas o dataset anexado, e treine utilizando AlexNet, VGG16 e ResNet18. apos isso me traga os resultados para eu avaliar os desempenhos e custos computacionais. e no final eu vou analisar os erros e limitacoes (para isso já me traga o necessario para eu analisar). o pdf já esta bem explicativo

  2. Política de IA:
     > nao sei se percebeu mas coloquei as politicas de uso da IA no trabalho

  3. Prompts de ajuda para o README (respostas às questões e limitações):
     > me mande tudo que eu preciso para responder as questoe, crie um md com as perguntas e tudo que eu preciso

     > respostas: de uma olhada se esta aceitavel. *(seguido de um rascunho das respostas 1.1 a 5.6 colado pelo grupo)*


     > as matriculas eu n vou preencher / e as limitacoes poderia dar um exemplo do que preencher? *(seguido da pergunta de Limitações do roteiro)*

     > eu achei que tudo esta certo, voce so falou demais, e nao teve tanta limitacao assim

     > sim

  4. Prompts operacionais: perguntas sobre o andamento do treino ("quanto tempo para acabar?", "falta quanto", "acabou?"), configuração do email do git e pedido de commit/push sem coautor.
- **Resposta/trecho gerado incorporado à entrega:** `src/data_pipeline.py`, `src/train_lib.py`, `tools/clean_notebook.py`, `pratica_watermeters.ipynb` (estrutura, células de aquisição, exploração, construção do dataset, treino, avaliação, figuras e diagnóstico de erros), `requirements.txt` e as seções factuais deste README (reprodução, protocolo, tabelas de resultados). A resposta da IA aqui está resumida, pois o código e os textos gerados são extensos e estão nos próprios arquivos do repositório.
- **Resultado do apoio da IA ao README (prompt 3):** (a) um guia local com a pergunta de cada questão, os números do projeto e a teoria envolvida; (b) uma revisão do rascunho do grupo, que apontou frases copiadas do guia, instruções soltas no texto e afirmações não verificadas; (c) o preenchimento dos trechos em colchetes do rascunho, incluindo duas contagens feitas pela IA olhando as imagens (classificação dos 16 erros compartilhados mais confiantes e cerca de 23 de 60 faixas com último dígito meio rolado), além da conferência do `classification_report` para a questão 5.1; (d) um exemplo de seção de Limitações, que o grupo encurtou. O texto final das respostas e das limitações no README incorpora esse material.
- **Adaptações realizadas pelo grupo:** o grupo conferiu tudo que a IA gerou, olhou os resultados dos treinamentos e checou se os números faziam sentido, incluindo as verificações obrigatórias do dataset e o split sem vazamento. As respostas das questões e as limitações foram escritas a partir de um rascunho do grupo, revisado e completado com apoio da IA (ver prompt 3 e o resultado acima); o grupo conferiu os números contra o `results.csv` e o notebook, revisou as contagens visuais nas galerias de erros e na auditoria das faixas, e finalizou o texto. Durante o desenvolvimento a IA corrigiu sozinha dois defeitos do próprio código, a cabeça que estava só no otimizador da `smallcnn` e impedia o aprendizado, e o pooling adaptativo que não é suportado no MPS, os dois detectados pela inspeção dos resultados.

**Seção B — Uso de skill (Claude Code / SKILL.md):** Não.
**Seção C — Uso de SDD:** Não.

**Declaração de responsabilidade:** Declaramos que as informações são verdadeiras e completas, que compreendemos e somos responsáveis por todo o conteúdo entregue, incluindo as partes geradas com auxílio de IA, e que estamos cientes de que a omissão do uso de IA configura conduta academicamente desonesta.
