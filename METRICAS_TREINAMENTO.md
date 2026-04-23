# Metricas de Treinamento (MLP)

Este arquivo explica as metricas e graficos que estamos plotando no projeto e como interpretar os sinais durante o treinamento.

## O que estamos plotando hoje

No fluxo atual, o `TrainingMonitor` plota:

1. `plot_loss()`
2. `plot_metric(metric_name="Accuracy" ou "R2")`
3. `plot_activation_histograms()`
4. `plot_gradient_histograms()`

## 1) Loss (treino e validacao)

### O que e
A funcao de custo otimizada pela rede.

- Classificacao MNIST: `CategoricalCrossEntropy`
- Regressao Boston: `MSE`

### O que observar
- `train_loss` caindo: modelo aprendendo no treino.
- `val_loss` caindo junto: generalizacao melhorando.
- `train_loss` cai e `val_loss` sobe: possivel overfitting.
- ambos altos e quase sem cair: underfitting, LR ruim, arquitetura fraca ou dados mal escalados.

### Sinais comuns
- Oscilacao forte da loss: learning rate pode estar alto.
- Loss muito lenta para cair: learning rate pode estar baixo.
- `val_loss` piora apos certo ponto: usar early stopping (se implementar).

## 2) Metrica principal (Accuracy ou R2)

## 2.1 Accuracy (MNIST)

### O que e
Proporcao de classes previstas corretamente.

### O que observar
- `train_acc` subindo e `val_acc` subindo: bom sinal.
- gap grande (`train_acc >> val_acc`): overfitting.
- acuracia estagnada perto de nivel baixo: capacidade insuficiente, LR ruim, ou gradiente ruim.

## 2.2 R2 (Boston)

### O que e
R2 mede o quanto o modelo explica a variancia do alvo.

- `R2 = 1`: ajuste perfeito.
- `R2 = 0`: tao bom quanto prever a media.
- `R2 < 0`: pior que prever a media.

### O que observar
- `train_r2` e `val_r2` subindo para valores positivos: bom progresso.
- `train_r2` alto e `val_r2` baixo: overfitting.
- `val_r2` negativo por muito tempo: modelo fraco, features insuficientes, LR/escala inadequados.

## 3) Histogramas de ativacoes

Mostram a distribuicao de saida (`A`) por camada ao longo das epocas.

### O que observar
- Distribuicao muito concentrada em um valor: saturacao ou baixa diversidade de representacao.
- ReLU com muita massa em zero: muitos neuronios "mortos" (dead ReLU).
- Ativacoes muito extremas: pode indicar instabilidade numerica ou escala ruim.

### Interpretacoes praticas
- Muitas ativacoes zeradas em ReLU: camada pode estar aprendendo pouco.
- Ativacoes todas muito parecidas: rede pode estar com baixa expressividade.

## 4) Histogramas de gradientes (dW)

Mostram distribuicao dos gradientes dos pesos por camada.

### O que observar
- Gradientes muito proximos de zero em varias camadas: vanishing gradient.
- Gradientes enormes (caudas longas): exploding gradient.
- Ultimas camadas com gradiente quase zero por muitas epocas: pouca atualizacao perto da saida.

### Exemplo: "gradientes em zero nas ultimas camadas"
Isso normalmente significa que essas camadas estao recebendo sinal de erro muito fraco para atualizar pesos.

Possiveis causas:
- learning rate muito baixo
- saturacao de ativacao
- inicializacao ruim
- arquitetura desbalanceada
- combinacao ativacao/loss pouco favoravel

Efeitos:
- aprendizado lento ou travado
- metrica estagna
- predicoes pouco melhoram entre epocas

Acoes praticas:
- testar LR um pouco maior
- revisar inicializacao (He para ReLU, Xavier para ativacoes simetricas)
- reduzir profundidade ou ajustar largura
- testar LeakyReLU em vez de ReLU em camadas problematicas
- verificar escala das features (especialmente em regressao)

## Padroes de diagnostico rapido

1. `train` melhora e `val` piora:
- overfitting.
- possiveis acoes: regularizacao, menos epocas, early stopping, mais dados.

2. `train` e `val` ruins:
- underfitting ou problema de otimizacao.
- possiveis acoes: mais capacidade, tuning de LR, melhor preprocessamento.

3. Loss oscila sem estabilizar:
- LR alto ou gradiente instavel.
- possiveis acoes: reduzir LR, revisar inicializacao.

4. Gradientes zeram cedo:
- vanishing/saturacao/dead ReLU.
- possiveis acoes: LeakyReLU, ajuste de inicializacao, tuning de arquitetura.

## Importante: interpretar em conjunto

Nenhum grafico isolado conta toda a historia.
A leitura mais confiavel vem da combinacao:

- loss + metrica + ativacoes + gradientes

Quando os quatro contam a mesma historia, o diagnostico fica bem mais confiavel.

## Demos praticas no projeto: vanishing e exploding

Agora o `main.py` tem dois modos didaticos para reproduzir os problemas:

1. `python main.py --mode vanishing`
2. `python main.py --mode exploding`

Ambos usam o dataset Boston para ficar rapido de executar.

## Como identificar no modo vanishing

Sinais tipicos esperados:

1. `loss` melhora pouco ou muito lentamente.
2. `R2` tende a estagnar em valor baixo.
3. No histograma de gradientes, varias camadas ficam muito concentradas perto de zero.
4. No resumo de gradientes impresso no terminal, `mean|dW|` e `||dW||2` ficam muito pequenos (por exemplo em notacao `e-08`, `e-10` etc, dependendo da rodada).

Leitura:

- O erro nao consegue propagar sinal forte ao longo da profundidade da rede.
- Como resultado, pesos quase nao mudam em parte da rede.

## Como identificar no modo exploding

Sinais tipicos esperados:

1. `loss` oscila fortemente ou diverge.
2. `R2` pode piorar rapido e ficar muito instavel.
3. Histograma de gradientes com caudas largas e valores muito altos.
4. Resumo de gradientes com `max|dW|` e `||dW||2` muito altos; em casos extremos pode aparecer `inf` ou `nan` na loss/metricas.

Leitura:

- As atualizacoes ficam grandes demais e o treino "sai do trilho".

## Regra de bolso para diagnostico rapido

1. Quase tudo perto de zero por muitas epocas: suspeite vanishing.
2. Valores muito grandes, explosivos, instabilidade numerica: suspeite exploding.
3. Use sempre os quatro sinais juntos: loss, metrica, ativacoes e gradientes.
