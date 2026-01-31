#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================
 SISTEMA DE REVERSÃO PRO - Análise Multi-Timeframe 1h + 5m
============================================================

ESTRATÉGIA:
  • TIMEFRAME 1h: Identifica zona de resistência/suporte com volume
  • TIMEFRAME 5m: Confirma entrada com Bollinger Bands + lateralização
  
FLUXO DE DETECÇÃO:
  1. Buscar candles 1h e 5m da API pública Binance
  2. Analisar contexto 1h (resistência/suporte + volume)
  3. Calcular Bollinger Bands no timeframe 5m
  4. Detectar lateralização após toque na banda
  5. Confirmar alinhamento entre timeframes
  6. Enviar alerta via Telegram (secrets protegidos)

SEGURANÇA:
  • API Binance: Pública (sem autenticação)
  • Telegram Token: Armazenado em Secrets do Replit
  • Nenhuma chave commitada no GitHub (.gitignore)
"""

import os
import time
import logging
from datetime import datetime, timedelta
import requests
import numpy as np
from telegram import Bot
from telegram.error import TelegramError
from dotenv import load_dotenv

# ============================================================================
# CONFIGURAÇÃO DE LOGGING - Registra todas as operações para auditoria
# ============================================================================
logging.basicConfig(
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    level=logging.INFO,
    datefmt='%d/%m %H:%M:%S',
    handlers=[
        logging.FileHandler("reversao_bot.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CARREGAR VARIÁVEIS DE AMBIENTE - Chaves vindas dos Secrets do Replit
# ============================================================================
load_dotenv()  # Carrega variáveis do ambiente (Secrets)

# Configurações sensíveis - VINDAS DOS SECRETS (nunca hardcoded!)
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')  # 🔒 Protegido em Secrets
CHAT_ID = os.getenv('CHAT_ID')                # 🔒 Protegido em Secrets

# Configurações públicas - Pode ser ajustado livremente
INTERVALO_VERIFICACAO = 300  # 5 minutos (alinhado com timeframe 5m)
PARES_MONITORADOS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']  # Personalize aqui!

# ============================================================================
# CLASSE: API BINANCE (Pública - Sem Autenticação Necessária)
# ============================================================================
class BinanceAPI:
    """Cliente para API pública da Binance - NENHUMA CHAVE NECESSÁRIA"""
    
    BASE_URL = "https://api.binance.com"
    
    @staticmethod
    def obter_klines(symbol: str, intervalo: str, limit: int = 100) -> list | None:
        """
        Obtém candles históricos da Binance (API Pública)
        
        Parâmetros:
            symbol: Par de negociação (ex: 'BTCUSDT')
            intervalo: Timeframe ('1h', '5m', '15m', etc.)
            limit: Número de candles a retornar (padrão: 100)
        
        Retorna:
            Lista de candles ou None em caso de erro
            
        Estrutura do candle [OHLCV]:
            [0] open_time: Timestamp de abertura
            [1] open: Preço de abertura
            [2] high: Máxima do candle
            [3] low: Mínima do candle
            [4] close: Preço de fechamento ← MAIS IMPORTANTE
            [5] volume: Volume negociado
            [6] close_time: Timestamp de fechamento
            ... (outros campos não usados)
        """
        try:
            url = f"{BinanceAPI.BASE_URL}/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': intervalo,
                'limit': limit
            }
            # Requisição pública - SEM headers de autenticação
            resposta = requests.get(url, params=params, timeout=10)
            resposta.raise_for_status()  # Lança exceção se status != 200
            return resposta.json()
        
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Erro de rede ao buscar {symbol} {intervalo}: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Erro inesperado ao buscar {symbol} {intervalo}: {e}")
            return None
    
    @staticmethod
    def formatar_preco(valor: float) -> str:
        """Formata valor monetário com padrão brasileiro (R$ 1.000,00)"""
        return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

# ============================================================================
# CLASSE: BOLLINGER BANDS - Cálculo Estatístico de Volatilidade
# ============================================================================
class BollingerBands:
    """Calcula as Bandas de Bollinger usando média móvel e desvio padrão"""
    
    @staticmethod
    def calcular(candles: list, periodo: int = 20, desvios: float = 2.0) -> dict | None:
        """
        Calcula Bollinger Bands
        
        Fórmula:
          SMA = Média móvel simples (20 períodos)
          Desvio Padrão = Volatilidade dos últimos 20 períodos
          Banda Superior = SMA + (2 × Desvio Padrão)
          Banda Inferior = SMA - (2 × Desvio Padrão)
          %B (Percent B) = (Preço - Banda Inferior) / (Banda Superior - Banda Inferior)
        
        Parâmetros:
            candles: Lista de candles OHLCV
            periodo: Período da média móvel (padrão: 20)
            desvios: Número de desvios padrão (padrão: 2.0)
        
        Retorna:
            Dicionário com bandas calculadas ou None se dados insuficientes
        """
        if len(candles) < periodo:
            logger.warning(f"⚠️ Dados insuficientes para BB ({len(candles)} < {periodo})")
            return None
        
        # Extrair preços de fechamento dos candles
        closes = np.array([float(c[4]) for c in candles])
        
        # Calcular média móvel simples (SMA)
        sma = np.convolve(closes, np.ones(periodo)/periodo, mode='valid')
        
        # Calcular desvio padrão móvel
        std = np.array([np.std(closes[i:i+periodo]) for i in range(len(closes) - periodo + 1)])
        
        # Calcular bandas
        superior = sma + (desvios * std)
        inferior = sma - (desvios * std)
        
        # Calcular %B (Percent B) - posição relativa do preço na banda
        # %B = 1.0 → tocou banda superior | %B = 0.0 → tocou banda inferior
        percent_b = (closes[periodo-1:] - inferior) / (superior - inferior)
        
        return {
            'superior': superior.tolist(),
            'media': sma.tolist(),
            'inferior': inferior.tolist(),
            'percent_b': percent_b.tolist(),
            'periodo': periodo,
            'desvios': desvios
        }

# ============================================================================
# CLASSE: DETECTOR DE REVERSÃO - Lógica Principal do Sistema
# ============================================================================
class DetectorReversao:
    """Detecta padrões de reversão com confirmação multi-timeframe"""
    
    @staticmethod
    def detectar_contexto_1h(klines_1h: list) -> dict | None:
        """
        Analisa contexto no timeframe 1h para identificar zonas de reversão
        
        CONDIÇÕES PARA RESISTÊNCIA (SINAL DE VENDA):
          1. Candle atual atingiu nova máxima significativa (+0.5% vs candle anterior)
          2. Candle fechou abaixo da máxima (-0.5% da máxima)
          3. Volume 20% acima da média dos últimos 20 candles
        
        CONDIÇÕES PARA SUPORTE (SINAL DE COMPRA):
          1. Candle atual atingiu nova mínima significativa (-0.5% vs candle anterior)
          2. Candle fechou acima da mínima (+0.5% da mínima)
          3. Volume 20% acima da média
        
        Retorna:
            Dicionário com detalhes da zona identificada ou None
        """
        if len(klines_1h) < 25:  # Precisa de 20 para média + 5 para análise
            return None
        
        # Últimos 5 candles para análise de reversão
        ultimos = klines_1h[-5:]
        
        # Dados do candle atual (último fechado)
        maxima_atual = float(ultimos[-1][2])   # high
        minima_atual = float(ultimos[-1][3])   # low
        fechamento_atual = float(ultimos[-1][4])  # close
        volume_atual = float(ultimos[-1][5])   # volume
        
        # Dados do candle anterior (para comparação)
        maxima_anterior = float(ultimos[-2][2])
        minima_anterior = float(ultimos[-2][3])
        
        # Calcular volume médio dos últimos 20 candles
        volumes_20 = [float(k[5]) for k in klines_1h[-20:]]
        volume_medio = np.mean(volumes_20)
        forca_volume = volume_atual / volume_medio  # > 1.0 = volume acima da média
        
        # ============================================================
        # DETECÇÃO DE RESISTÊNCIA (Potencial SINAL DE VENDA)
        # ============================================================
        if (maxima_atual > maxima_anterior * 1.005 and    # Nova máxima +0.5%
            fechamento_atual < maxima_atual * 0.995 and    # Fechou -0.5% da máxima
            forca_volume > 1.2):                          # Volume 20% acima da média
            
            logger.info(f"🔍 Resistência detectada em 1h: R$ {BinanceAPI.formatar_preco(maxima_atual)} "
                       f"(volume {forca_volume:.2f}x)")
            
            return {
                'tipo': 'resistencia',
                'preco_zona': maxima_atual,
                'preco_atual': fechamento_atual,
                'forca_volume': forca_volume,
                'timestamp': int(ultimos[-1][0]),
                'timeframe': '1h'
            }
        
        # ============================================================
        # DETECÇÃO DE SUPORTE (Potencial SINAL DE COMPRA)
        # ============================================================
        if (minima_atual < minima_anterior * 0.995 and    # Nova mínima -0.5%
            fechamento_atual > minima_atual * 1.005 and    # Fechou +0.5% da mínima
            forca_volume > 1.2):                          # Volume acima da média
            
            logger.info(f"🔍 Suporte detectado em 1h: R$ {BinanceAPI.formatar_preco(minima_atual)} "
                       f"(volume {forca_volume:.2f}x)")
            
            return {
                'tipo': 'suporte',
                'preco_zona': minima_atual,
                'preco_atual': fechamento_atual,
                'forca_volume': forca_volume,
                'timestamp': int(ultimos[-1][0]),
                'timeframe': '1h'
            }
        
        return None  # Nenhuma zona de reversão detectada
    
    @staticmethod
    def detectar_entrada_5m(klines_5m: list, bb: dict) -> dict | None:
        """
        Analisa entrada no timeframe 5m com Bollinger Bands
        
        CONDIÇÕES PARA LATERALIZAÇÃO APÓS TOQUE:
          1. Último candle fechado tocou banda (%B > 0.95 ou < 0.05)
          2. 2-3 candles seguintes com amplitude pequena (< 0.3%)
          3. Preço consolidando dentro da banda (não rompendo)
        
        Retorna:
            Dicionário com detalhes da entrada ou None
        """
        if len(klines_5m) < 15 or bb is None:
            return None
        
        # Percentual B do último candle FECHADO (-2 porque -1 é candle em andamento)
        percent_b = bb['percent_b'][-2]
        
        # Calcular amplitude dos últimos 3 candles fechados
        amplitudes = []
        for i in range(-4, -1):  # candles -4, -3, -2 (3 candles fechados)
            high = float(klines_5m[i][2])
            low = float(klines_5m[i][3])
            open_price = float(klines_5m[i][1])
            amplitude = (high - low) / open_price  # amplitude relativa %
            amplitudes.append(amplitude)
        
        amplitude_media = np.mean(amplitudes)
        
        # ============================================================
        # LATERALIZAÇÃO APÓS TOQUE NA BANDA SUPERIOR (VENDA)
        # ============================================================
        if (percent_b > 0.95 and           # Toque na banda superior (%B > 95%)
            amplitude_media < 0.003):      # Amplitude média < 0.3%
            
            logger.info(f"📊 Lateralização detectada em 5m após toque superior "
                       f"(%B={percent_b:.2f}, amp={amplitude_media:.4f})")
            
            return {
                'tipo': 'lateralizacao_superior',
                'percent_b': percent_b,
                'amplitude_media': amplitude_media,
                'preco_atual': float(klines_5m[-2][4]),  # fechamento último candle
                'timestamp': int(klines_5m[-2][0]),
                'timeframe': '5m'
            }
        
        # ============================================================
        # LATERALIZAÇÃO APÓS TOQUE NA BANDA INFERIOR (COMPRA)
        # ============================================================
        if (percent_b < 0.05 and           # Toque na banda inferior (%B < 5%)
            amplitude_media < 0.003):      # Amplitude média < 0.3%
            
            logger.info(f"📊 Lateralização detectada em 5m após toque inferior "
                       f"(%B={percent_b:.2f}, amp={amplitude_media:.4f})")
            
            return {
                'tipo': 'lateralizacao_inferior',
                'percent_b': percent_b,
                'amplitude_media': amplitude_media,
                'preco_atual': float(klines_5m[-2][4]),
                'timestamp': int(klines_5m[-2][0]),
                'timeframe': '5m'
            }
        
        return None  # Nenhuma lateralização detectada
    
    @staticmethod
    def confirmar_sinal(symbol: str) -> dict | None:
        """
        Confirmação completa do sinal com alinhamento multi-timeframe
        
        FLUXO:
          1. Buscar dados 1h → Analisar contexto (resistência/suporte)
          2. Buscar dados 5m → Calcular BB + detectar lateralização
          3. Validar alinhamento entre timeframes
          4. Retornar sinal confirmado
        
        Retorna:
            Dicionário com sinal completo ou None se não confirmado
        """
        logger.info(f"🔍 Iniciando análise de {symbol}...")
        
        # Passo 1: Buscar candles 1h
        klines_1h = BinanceAPI.obter_klines(symbol, '1h', limit=50)
        if not klines_1h:
            logger.warning(f"⚠️ Dados 1h indisponíveis para {symbol}")
            return None
        
        # Passo 2: Buscar candles 5m
        klines_5m = BinanceAPI.obter_klines(symbol, '5m', limit=50)
        if not klines_5m:
            logger.warning(f"⚠️ Dados 5m indisponíveis para {symbol}")
            return None
        
        # Passo 3: Calcular Bollinger Bands 5m
        bb_5m = BollingerBands.calcular(klines_5m, periodo=20, desvios=2.0)
        if not bb_5m:
            logger.warning(f"⚠️ BB não calculado para {symbol} 5m")
            return None
        
        # Passo 4: Detectar contexto 1h
        contexto = DetectorReversao.detectar_contexto_1h(klines_1h)
        if not contexto:
            logger.debug(f"ℹ️ Nenhum contexto de reversão em 1h para {symbol}")
            return None
        
        # Passo 5: Detectar entrada 5m
        entrada = DetectorReversao.detectar_entrada_5m(klines_5m, bb_5m)
        if not entrada:
            logger.debug(f"ℹ️ Nenhuma lateralização detectada em 5m para {symbol}")
            return None
        
        # Passo 6: CONFIRMAÇÃO - Alinhamento entre timeframes
        # ============================================================
        # SINAL DE VENDA: Resistência 1h + Lateralização Superior 5m
        # ============================================================
        if (contexto['tipo'] == 'resistencia' and 
            entrada['tipo'] == 'lateralizacao_superior'):
            
            # Calcular risco percentual (distância da zona de resistência)
            risco = (contexto['preco_zona'] - entrada['preco_atual']) / contexto['preco_zona']
            
            sinal = {
                'acao': 'VENDA',
                'symbol': symbol,
                'preco_entrada': entrada['preco_atual'],
                'preco_zona': contexto['preco_zona'],
                'risco_percentual': abs(risco) * 100,
                'forca_volume': contexto['forca_volume'],
                'timestamp': datetime.now().strftime('%d/%m %H:%M:%S'),
                'timeframe_contexto': '1h',
                'timeframe_entrada': '5m',
                'confianca': 'ALTA' if contexto['forca_volume'] > 1.5 else 'MÉDIA'
            }
            
            logger.info(f"✅ SINAL CONFIRMADO | {sinal['acao']} | {symbol} | "
                       f"R$ {BinanceAPI.formatar_preco(sinal['preco_entrada'])}")
            return sinal
        
        # ============================================================
        # SINAL DE COMPRA: Suporte 1h + Lateralização Inferior 5m
        # ============================================================
        if (contexto['tipo'] == 'suporte' and 
            entrada['tipo'] == 'lateralizacao_inferior'):
            
            risco = (entrada['preco_atual'] - contexto['preco_zona']) / contexto['preco_zona']
            
            sinal = {
                'acao': 'COMPRA',
                'symbol': symbol,
                'preco_entrada': entrada['preco_atual'],
                'preco_zona': contexto['preco_zona'],
                'risco_percentual': abs(risco) * 100,
                'forca_volume': contexto['forca_volume'],
                'timestamp': datetime.now().strftime('%d/%m %H:%M:%S'),
                'timeframe_contexto': '1h',
                'timeframe_entrada': '5m',
                'confianca': 'ALTA' if contexto['forca_volume'] > 1.5 else 'MÉDIA'
            }
            
            logger.info(f"✅ SINAL CONFIRMADO | {sinal['acao']} | {symbol} | "
                       f"R$ {BinanceAPI.formatar_preco(sinal['preco_entrada'])}")
            return sinal
        
        # Timeframes não alinhados - sinal rejeitado
        logger.debug(f"ℹ️ Timeframes não alinhados para {symbol} "
                    f"({contexto['tipo']} vs {entrada['tipo']})")
        return None

# ============================================================================
# CLASSE: ALERTA TELEGRAM - Envio Seguro de Notificações
# ============================================================================
class TelegramAlerta:
    """Envia alertas formatados para Telegram com proteção contra spam"""
    
    def __init__(self):
        # Inicializar bot com token vindo dos Secrets (🔒 protegido)
        if not TELEGRAM_TOKEN:
            raise ValueError("❌ TELEGRAM_TOKEN não configurado nos Secrets!")
        
        self.bot = Bot(token=TELEGRAM_TOKEN)
        self.alertas_enviados = set()  # Evitar alertas duplicados
    
    def enviar_sinal(self, sinal: dict) -> bool:
        """
        Envia alerta de sinal formatado para Telegram
        
        Proteção contra spam:
          - Um sinal por par por dia
          - Chave única: SYMBOL_ACAO_DATA
        
        Retorna:
            True se enviado com sucesso, False caso contrário
        """
        # Chave única para evitar duplicatas (ex: "BTCUSDT_VENDA_31/01")
        chave_unica = f"{sinal['symbol']}_{sinal['acao']}_{sinal['timestamp'][:5]}"
        
        if chave_unica in self.alertas_enviados:
            logger.debug(f"ℹ️ Alerta duplicado ignorado: {chave_unica}")
            return False
        
        # Emojis e cores para formatação
        emoji = "🔻" if sinal['acao'] == 'VENDA' else "🟢"
        acao_fmt = "VENDA 📉" if sinal['acao'] == 'VENDA' else "COMPRA 📈"
        
        # Mensagem formatada em Markdown
        mensagem = (
            f"{emoji} *SINAL DE REVERSÃO CONFIRMADO* {emoji}\n\n"
            f"{'═' * 35}\n"
            f"🪙 *{sinal['symbol'].replace('USDT', '/USDT')}*\n"
            f"📊 *AÇÃO:* {acao_fmt}\n"
            f"💰 *Entrada:* {BinanceAPI.formatar_preco(sinal['preco_entrada'])}\n"
            f"🎯 *Zona:* {BinanceAPI.formatar_preco(sinal['preco_zona'])}\n"
            f"{'═' * 35}\n\n"
            f"🔍 *CONFIRMAÇÃO MULTI-TIMEFRAME*\n"
            f"   • Contexto (`1h`): Zona de {sinal['acao'].lower()} identificada\n"
            f"   • Entrada (`5m`): Bollinger Bands + lateralização\n"
            f"   • Volume: `{sinal['forca_volume']:.2f}x` média (força {sinal['confianca']})\n"
            f"   • Risco estimado: `{sinal['risco_percentual']:.2f}%`\n\n"
            f"⚠️ *GESTÃO DE RISCO (IMPORTANTE)*\n"
            f"   • Stop Loss: {'2% acima' if sinal['acao'] == 'VENDA' else '2% abaixo'} da entrada\n"
            f"   • Take Profit: Relação 1:2 (risco:retorno)\n"
            f"   • Alavancagem máxima recomendada: 3x\n"
            f"   • Risco por operação: ≤ 2% do capital\n\n"
            f"⏰ *Hora do Sinal:* {sinal['timestamp']} (BRT)\n"
            f"📡 *Sistema:* Reversão Pro v1.0"
        )
        
        try:
            # Enviar mensagem para o chat ID configurado nos Secrets
            if not CHAT_ID:
                raise ValueError("CHAT_ID não configurado nos Secrets!")
            
            self.bot.send_message(
                chat_id=int(CHAT_ID),  # Converter para inteiro
                text=mensagem,
                parse_mode='Markdown',
                disable_web_page_preview=True
            )
            
            # Registrar alerta enviado
            self.alertas_enviados.add(chave_unica)
            
            # Salvar no histórico para auditoria
            with open('historico_sinais.log', 'a', encoding='utf-8') as f:
                f.write(f"{datetime.now().isoformat()} | {sinal}\n")
            
            logger.info(f"✅ Alerta enviado para Telegram: {sinal['acao']} {sinal['symbol']}")
            return True
            
        except TelegramError as e:
            logger.error(f"❌ Erro Telegram: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Erro inesperado ao enviar alerta: {e}")
            return False

# ============================================================================
# CLASSE: BOT PRINCIPAL - Loop de Execução 24/7
# ============================================================================
class ReversaoBot:
    """Orquestrador principal do sistema de detecção de reversões"""
    
    def __init__(self):
        self.alerta = TelegramAlerta()
        self.ultima_verificacao = {}  # Controle de rate limiting por par
        logger.info("✅ Bot de Reversão Pro inicializado")
        logger.info(f"📊 Pares monitorados: {', '.join(PARES_MONITORADOS)}")
        logger.info(f"⏱️ Intervalo de verificação: {INTERVALO_VERIFICACAO}s")
    
    def executar(self):
        """Loop principal de verificação contínua"""
        logger.info("🚀 Iniciando detecção de reversões 24/7...")
        logger.info("ℹ️ Sistema aguardando condições de reversão...")
        
        while True:
            ciclo_inicio = datetime.now()
            
            for symbol in PARES_MONITORADOS:
                # Rate limiting: verificar cada par a cada 5 minutos
                ultima = self.ultima_verificacao.get(symbol, ciclo_inicio - timedelta(minutes=6))
                if (datetime.now() - ultima).total_seconds() < INTERVALO_VERIFICACAO:
                    continue
                
                logger.info(f"🔍 Analisando {symbol}...")
                
                try:
                    # Detectar e confirmar sinal
                    sinal = DetectorReversao.confirmar_sinal(symbol)
                    
                    # Enviar alerta se sinal confirmado
                    if sinal:
                        self.alerta.enviar_sinal(sinal)
                    
                    # Registrar última verificação
                    self.ultima_verificacao[symbol] = datetime.now()
                    
                except Exception as e:
                    logger.error(f"💥 Erro crítico ao analisar {symbol}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
            
            # Calcular tempo de espera real (compensar tempo de processamento)
            ciclo_duracao = (datetime.now() - ciclo_inicio).total_seconds()
            espera = max(1, INTERVALO_VERIFICACAO - ciclo_duracao)
            
            logger.info(f"😴 Aguardando {espera:.0f}s para próximo ciclo...")
            time.sleep(espera)

# ============================================================================
# PONTO DE ENTRADA PRINCIPAL
# ============================================================================
def main():
    """Função principal - validação de configuração e inicialização"""
    
    # Validar configuração de segurança
    if not TELEGRAM_TOKEN:
        logger.critical("❌ ERRO CRÍTICO: TELEGRAM_TOKEN não configurado!")
        logger.critical("   Configure em Replit: Tools → Secrets")
        logger.critical("   NUNCA commitar chaves reais no GitHub!")
        return
    
    if not CHAT_ID:
        logger.critical("❌ ERRO CRÍTICO: CHAT_ID não configurado!")
        logger.critical("   Configure em Replit: Tools → Secrets")
        return
    
    logger.info("✅ Configuração de segurança validada")
    logger.info(f"   • Telegram Token: {'*' * 8}{TELEGRAM_TOKEN[-4:]}")  # Mostrar só últimos 4 dígitos
    logger.info(f"   • Chat ID: {CHAT_ID}")
    
    # Iniciar bot
    try:
        bot = ReversaoBot()
        bot.executar()
    except KeyboardInterrupt:
        logger.info("🛑 Bot interrompido manualmente pelo usuário")
    except Exception as e:
        logger.critical(f"💥 Erro fatal não tratado: {e}")
        import traceback
        logger.critical(traceback.format_exc())

if __name__ == "__main__":
    main()
