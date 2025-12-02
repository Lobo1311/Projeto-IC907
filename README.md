# Projeto-IC907
Implementação de uma Rede Neural (NN do inglês _Neural Network_) utilizando Python. 
Implementação de uma PINN utilizando código baseado em PyTorch.

## Sumário
- [Sobre]
- [Como utilizar]
- [Dependências]

## Sobre
Este projeto tem como objetivo a implementação de uma ANN completa utilizando Python.
Tem-se como pré-requisito a instalação da biblioteca Numpy.

O código implementado possui suporte para os seguintes:
- Função de ativação: ReLU, LeakyReLU, TanH, Sigmoide;
- Erro: Média dos quadrados dos erros (Mean Squared Error - MSE);
- Otimização: Método dos gradientes descendentes (Gradient descent).

Além disso, foi utilizada a biblioteca PyTorch para criação de uma PINN para solução do problema de Darcy em fluxo transiente em 1D.
Para utilização do PINN o PyTorch deve ser instalado.

Na seção seguinte é mostrada como utilizar o código, para a rede ANN desenvolvida do zero e para a PINN.

## Como utilizar

Para utilização do código, o arquivo Main.py deve ser utilizado. Deve-se seguir o seguinte procedimento:

1 . No arquivo Main.py, encontrar o condicional <if __name__ == "__main__":> e definir o tipo de treinamento desejado.
  1.1 Para utilização da rede neural desenvolvida do zero, chamar o método <main_nn_by_hand()>.
  1.2 Para utilização da PINN, chamar o método <main_nn_torch()>;
2. Se rede neural do zero:
  2.1 Procurar por "#* Hyperparameters definition" para encontrar os parâmetros passiveis de alteração (estes podem ser alterados conforme valores apresentados em relatório para reprodução de resultados).
  2.2 NOTA: O restante do método <main_nn_by_hand()> não precisa ser alterado.
3. Se PINN:
  3.1 Procurar por "#* problem definition" para alteração dos parâmetros do problema de Darcy (não devem ser alterados para reprodução de resultados).
  3.2 Procurar por "#* Hyperparameters definition" para encontrar os parâmetros passiveis de alteração (estes podem ser alterados conforme valores apresentados em relatório para reprodução de resultados).
  3.3 NOTA: O restante do método <main_nn_torch()> não precisa ser alterado.


## Dependências

O código aqui apresentado necessita de duas dependências para funcionamento completo. Sua especificação e como instalar são apresentadas abaixo:

1. Numpy: Para instalção do Numpy, abra o CMD/PowerShell e utilize o comando "pip install numpy".
2. PyTorch: Para instalação do PyTorch, abra o CMD/PowerShell e utilize o comando "pip install torch torchvision torchaudio".
