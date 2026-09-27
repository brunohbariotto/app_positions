# Otimização das janelas dos osciladores

No menu **Otimização de Janelas**, selecione ativos, grade de períodos e custos.
Execute, examine o teste separado e salve os ativos desejados. A aplicação cria/
atualiza a aba `Oscillator_Windows` da planilha `Position_Control`, preservando
os outros ativos já salvos e as demais abas. Para aplicar, marque **Usar janelas
otimizadas por ativo** em **Alocação por Ativo**. Desmarcada, a execução original
permanece intacta; a fórmula em `models.py` não foi modificada.

## O que os períodos significam

`slow_dias`, `medium_dias` e `fast_dias` representam a constante de tempo maior
de cada par EWM, hoje 96, 48 e 24. A constante menor é um terço desses valores
(32, 16, 8); o código legado usa `span = 2 * constante - 1`. São observações
diárias de mercado, sem criar preços para feriados. Criptomoedas têm observações
também nos fins de semana. A grade exige múltiplos de três e Fast < Medium < Slow.
A busca inclui sempre a combinação padrão e encontra o melhor candidato dentre
os avaliados, sem pretender identificar um ótimo universal.

Não se otimiza o tamanho do histórico usado para os quartis: eles são expansivos,
usam somente dados anteriores à decisão e exigem 64 observações. Os percentuais
por quartil são os mesmos do legado, incluindo a exposição máxima de 125%.

## Método e comparação

- Yahoo `period=max`, preços ajustados, um download por ativo, cache de seis horas.
  A barra do dia corrente é excluída. Não se corta o histórico na data de início
  dos outros ativos, não há preenchimento retrospectivo e não há substituição
  silenciosa de ETFs por índices ou criptomoedas.
- A fórmula de sinal mantém EWM, volatilidade, tanh e limites do legado, mas usa
  a normalização para um único ativo. Isso torna as janelas independentes da cesta.
- Aquecimento comum: três vezes a maior janela Slow da grade. São exigidas
  pelo menos 315 observações adicionais, com mínimo de 63 por partição.
- Treino nos primeiros 60% após aquecimento: ranking por retorno líquido.
  Validação nos próximos 20%: escolhe entre os dez melhores e o padrão.
  Teste nos últimos 20%: somente avaliação, sem alterar a escolha.
- Sinal no fechamento t, execução no fechamento t+1; primeiro retorno recebido
  termina em t+2. O custo incide sobre a mudança absoluta de exposição. A carteira
  simulada mantém suas posições entre as divisões cronológicas. Comprar e manter
  inclui custo de entrada no começo de cada período comparativo.
- O acerto direcional mede mudanças da alocação versus o retorno desde a execução
  até o horizonte escolhido. O horizonte é diagnóstico, não objetivo da busca.
- “Padrão causal” é 96/48/24 no mesmo motor individual, com o mesmo histórico e
  execução. Não equivale ao backtest do legado, que usa quartis da amostra inteira
  e normalização conjunta dos ativos. Os gráficos legados na tela de alocação
  permanecem identificados; a tabela posterior informa os percentuais aplicados.

As métricas são frações, por exemplo 0,1 = 10%. A simulação é de exposição
percentual: não modela mudanças do Markowitz, impostos, financiamento da exposição
acima de 100%, remuneração do caixa, lotes ou quantidades fixas de ações. O teste
pode perder para o padrão; isso é mostrado e não é usado para reescolher janelas.
Repetir pesquisas olhando o teste também pode causar sobreajuste.

`Taxa_Juros_Brasil` continua com sua regra original, pois uma taxa FRED não é um
preço de investimento. Para otimizá-la será necessário definir um ativo ou índice
de retorno correspondente e tratar a disponibilidade histórica das publicações.

## Persistência e aplicação

A aba armazena ticker, três janelas, versão do motor, datas, tamanho do histórico,
grade, custos, aquecimento, frações de validação/teste e métricas. O salvamento
ocorre apenas no botão da página; a execução da busca não grava a planilha.
A conexão usa as credenciais Google que a aplicação já utiliza.

Na aplicação, cada ativo configurado é recalculado com o mesmo motor e todo o
histórico até a data final selecionada. A data inicial do menu não limita esse
histórico. Configuração ausente/inválida, falhas de download ou histórico curto
mantêm o percentual original com aviso. Não se aplicam janelas pesquisadas com
dados posteriores a uma data final histórica selecionada. Uma configuração
antiga pode continuar ativa; as datas da pesquisa e da última cotação ficam visíveis.

Para evitar conflitos, não salvar simultaneamente a mesma aba em duas sessões:
a API atual faz leitura e atualização, sem transação de escrita concorrente.

## Verificação

`python -m pytest tests -q`

Os testes verificam equivalência do sinal individual ao legado, pesos, ausência de
vazamento temporal, execução/custos, separação do teste, persistência, fallback e
o fluxo Streamlit de pesquisar, revisar e salvar com uma planilha simulada.

Referências: [avaliação temporal do scikit-learn](https://sklearn.org/stable/modules/cross_validation.html#time-series-split)
e [histórico yfinance](https://ranaroussi.github.io/yfinance/reference/api/yfinance.Ticker.html).
