import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io # Necessário para ler o uploaded_file como buffer
from typing import Dict, Any, Optional, Tuple # Importar Tuple para type hints

# Importar biblioteca HPLC
# Verificar se a biblioteca está disponível no início
HPLC_LIB_AVAILABLE = False
try:
    # Tentativa de importar as classes e funções necessárias
    from hplc.quant import Chromatogram
    # hplc.io.load_chromatogram não é estritamente necessário se carregarmos com pandas
    # from hplc.io import load_chromatogram
    HPLC_LIB_AVAILABLE = True
except ImportError:
    st.error("⚠️ Biblioteca HPLC não encontrada. Por favor instale: pip install hplc")
    st.info("⚠️ Algumas funcionalidades (detecção e análise de picos) não estarão disponíveis sem a biblioteca HPLC.")

def convert_time_to_minutes(data: pd.DataFrame, time_col: str, time_unit: str) -> pd.DataFrame:
    """
    Converter coluna de tempo para minutos.

    Args:
        data: DataFrame pandas contendo os dados do cromatograma.
        time_col: Nome da coluna de tempo.
        time_unit: Unidade atual da coluna de tempo ("Segundos", "Minutos", "Horas").

    Returns:
        DataFrame com a coluna de tempo convertida para minutos.
    """
    # Usamos .loc para evitar SettingWithCopyWarning
    data_copy = data.copy()

    if time_unit == "Segundos":
        data_copy.loc[:, time_col] = data_copy[time_col].apply(lambda x: x / 60)
        st.info("✅ Tempo convertido de segundos para minutos")
    elif time_unit == "Milisegundos":
        data_copy.loc[:, time_col] = data_copy[time_col].apply(lambda x: x / 60000)
        st.info("✅ Tempo convertido de milissegundos para minutos")
    elif time_unit == "Minutos":
        st.info("✅ Tempo já está em minutos")
    # Adicionar um else para unidades desconhecidas, embora o selectbox limite as opções
    else:
         st.warning(f"Unidade de tempo desconhecida: {time_unit}. Nenhuma conversão aplicada.")


    return data_copy

# Nova função para processar com a biblioteca HPLC
def process_chromatogram_hplc(data_df: pd.DataFrame, time_col: str, signal_col: str, params: Dict[str, Any]) -> Tuple[Optional[Chromatogram], pd.DataFrame]:
    """
    Processar cromatograma usando a biblioteca HPLC (criação de objeto Chromatogram e fit_peaks).

    Args:
        data_df: DataFrame pandas contendo os dados processados (tempo em minutos).
        time_col: Nome da coluna de tempo.
        signal_col: Nome da coluna de sinal.
        params: Dicionário de parâmetros para a função fit_peaks.

    Returns:
        Uma tupla contendo o objeto Chromatogram (ou None se HPLC_LIB_AVAILABLE for False)
        e um DataFrame com os dados dos picos (vazio se não houver picos ou ocorrer erro).
    """
    if not HPLC_LIB_AVAILABLE:
        st.error("❌ Biblioteca HPLC não está disponível. Não é possível processar picos.")
        return None, pd.DataFrame()

    try:
        st.info("🛠️ Processando com biblioteca HPLC...")
        # Criar objeto Chromatogram. A biblioteca espera o DataFrame e um dict de colunas.
        # Certificar-se que os nomes das colunas passados existem no DataFrame
        if time_col not in data_df.columns or signal_col not in data_df.columns:
             st.error(f"Colunas '{time_col}' ou '{signal_col}' não encontradas no DataFrame.")
             return None, pd.DataFrame()

        cromatograma = Chromatogram(data_df, cols={'time': time_col, 'signal': signal_col})

        # Aplicar fit_peaks com os parâmetros
        # fit_peaks retorna um DataFrame com os picos encontrados
        dados_picos = cromatograma.fit_peaks(
            correct_baseline=params['correct_baseline'],
            approx_peak_width=params['approx_peak_width'],
            buffer=params['buffer'],
            prominence=params['prominence']
        )

        st.success("✅ Análise de picos concluída.")
        return cromatograma, dados_picos

    except Exception as e:
        st.error(f"❌ Erro ao processar com biblioteca HPLC: {str(e)}")
        st.exception(e)
        return None, pd.DataFrame()

def main():
    st.set_page_config(
        page_title="Analisador de Cromatogramas",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 Analisador de Cromatogramas")
    st.markdown("---")

    # Sidebar para parâmetros
    st.sidebar.header("⚙️ Parâmetros de Análise (HPLC Lib)")

    # Mostrar parâmetros apenas se a biblioteca HPLC estiver disponível
    params: Dict[str, Any] = {}
    if HPLC_LIB_AVAILABLE:
        params['correct_baseline'] = st.sidebar.checkbox("Corrigir linha de base", value=False, key="correct_baseline_checkbox")
        params['approx_peak_width'] = st.sidebar.slider("Largura aproximada do pico", 0.01, 5.0, 0.1, 0.01, key="peak_width_slider", help="Largura esperada dos picos em unidades de tempo (minutos).")
        params['buffer'] = st.sidebar.slider("Buffer", 10, 500, 100, 10, key="buffer_slider", help="Número de pontos à volta de um pico para considerar na correção da linha de base.")
        params['prominence'] = st.sidebar.slider("Prominência", 0.001, 1.0, 0.02, 0.001, key="prominence_slider", help="Proeminência mínima para detetar um pico.")
    else:
         st.sidebar.warning("Parâmetros de análise desabilitados (biblioteca HPLC não encontrada).")
         # Definir parâmetros padrão ou vazios se a lib não estiver disponível
         params = {
             'correct_baseline': False,
             'approx_peak_width': 0.1,
             'buffer': 100,
             'prominence': 0.02
         }


    # Upload de ficheiro
    st.header("📁 Carregar Ficheiro")
    uploaded_file = st.file_uploader(
        "Escolha um ficheiro CSV ou DAT",
        type=['csv', 'dat'],
        help="Carregue um ficheiro contendo dados de cromatografia. O ficheiro deve ter pelo menos colunas para Tempo e Sinal.",
        key="file_uploader"
    )

    data_df: Optional[pd.DataFrame] = None # Inicializa DataFrame como None

    if uploaded_file is not None:
        # Ler o ficheiro para um buffer em memória para poder lê-lo várias vezes (preview, load total)
        file_buffer = io.BytesIO(uploaded_file.getvalue())

        # Detectar delimitador
        delimiter = st.selectbox(
            "Delimitador",
            [',', ';', '\t', ' '],
            index=0, # Pode tentar adivinhar o index inicial com base no nome do ficheiro ou conteúdo se quiser
            key="delimiter_selector",
            help="Escolha o delimitador usado no ficheiro"
        )

        try:
            # Primeiro, ler um pouco do ficheiro para mostrar preview e selecionar colunas
            # Usamos BytesIO e seek(0) para garantir que a leitura começa do início
            file_buffer.seek(0)
            # Tenta ler com o delimitador selecionado
            try:
                 # pd.read_csv pode ter problemas com encoding, usar 'latin1' ou 'cp1252' é comum para dados de instrumentos
                 data_preview = pd.read_csv(file_buffer, delimiter=delimiter, nrows=10, encoding='utf-8')
            except UnicodeDecodeError:
                 file_buffer.seek(0) # Reset novamente para tentar outra encoding
                 data_preview = pd.read_csv(file_buffer, delimiter=delimiter, nrows=10, encoding='latin1')


            st.success(f"✅ Ficheiro carregado com sucesso! (Lidas primeiras 10 linhas para pré-visualização)")

            # Mostrar preview dos dados
            with st.expander("🔍 Pré-visualização dos dados e Informações"):
                st.dataframe(data_preview)

                # Tentar ler o ficheiro completo para obter o número total de linhas e colunas
                file_buffer.seek(0)
                try:
                    full_data_info = pd.read_csv(file_buffer, delimiter=delimiter, encoding='utf-8')
                except UnicodeDecodeError:
                    file_buffer.seek(0)
                    full_data_info = pd.read_csv(file_buffer, delimiter=delimiter, encoding='latin1')


                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Informações do dataset:**")
                    st.write(f"- Nome do ficheiro: {uploaded_file.name}")
                    st.write(f"- Tamanho do ficheiro: {uploaded_file.size} bytes")
                    st.write(f"- Número de linhas (estimado): {full_data_info.shape[0]}")
                    st.write(f"- Número de colunas: {full_data_info.shape[1]}")
                    st.write(f"- Colunas: {list(full_data_info.columns)}")

                with col2:
                    st.write("**Estatísticas básicas (primeiras 10 linhas):**")
                    try:
                        st.write(data_preview.describe())
                    except Exception as e:
                         st.warning(f"Não foi possível gerar estatísticas básicas: {e}")


            # Selecionar colunas e unidade de tempo
            st.header("📋 Configuração das Colunas")

            available_columns = full_data_info.columns.tolist() # Usar colunas do ficheiro completo

            if not available_columns:
                 st.error("Não foram detetadas colunas no ficheiro. Verifique o delimitador.")
                 st.stop() # Para a execução se não há colunas

            col1, col2, col3 = st.columns(3)

            # Tentar pré-selecionar colunas comuns
            default_time_col_index = 0
            if 'time' in available_columns:
                 default_time_col_index = available_columns.index('time')
            elif 'tempo' in available_columns:
                 default_time_col_index = available_columns.index('tempo')

            default_signal_col_index = 1 if len(available_columns) > 1 else 0
            if 'signal' in available_columns:
                 default_signal_col_index = available_columns.index('signal')
            elif 'sinal' in available_columns:
                 default_signal_col_index = available_columns.index('sinal')
            elif 'intensity' in available_columns:
                 default_signal_col_index = available_columns.index('intensity')


            with col1:
                time_col = st.selectbox(
                    "Coluna de Tempo",
                    available_columns,
                    index=default_time_col_index,
                    key="time_column_selector",
                    help="Selecione a coluna que contém os dados de tempo"
                )

            with col2:
                signal_col = st.selectbox(
                    "Coluna de Sinal",
                    available_columns,
                    index=default_signal_col_index,
                    key="signal_column_selector",
                    help="Selecione a coluna que contém os dados de sinal"
                )

            with col3:
                time_unit = st.selectbox(
                    "Unidade de Tempo",
                    ["Segundos", "Minutos", "Milissegundos"],
                    index=0, # Assume segundos como padrão, pode ajustar
                    key="time_unit_selector",
                    help="Escolha a unidade da coluna de tempo (será convertida para minutos)"
                )

            # Botão para processar
            if st.button("🚀 Processar Cromatograma", type="primary", key="process_button"):
                if not time_col or not signal_col:
                     st.warning("Por favor, selecione as colunas de Tempo e Sinal.")
                else:
                    with st.spinner("Carregando e processando dados..."):
                        try:
                            # Carregar o ficheiro completo usando o delimitador selecionado
                            file_buffer.seek(0) # Reset buffer position
                            try:
                                data_df = pd.read_csv(file_buffer, delimiter=delimiter, encoding='utf-8')
                            except UnicodeDecodeError:
                                file_buffer.seek(0)
                                data_df = pd.read_csv(file_buffer, delimiter=delimiter, encoding='latin1')

                            # Converter tempo para minutos
                            data_df = convert_time_to_minutes(data_df, time_col, time_unit)

                            # --- Etapa de Processamento (se a biblioteca HPLC estiver disponível) ---
                            cromatograma = None
                            dados_picos = pd.DataFrame()

                            if HPLC_LIB_AVAILABLE:
                                # Chamar a função de processamento HPLC
                                cromatograma, dados_picos = process_chromatogram_hplc(data_df, time_col, signal_col, params)
                            else:
                                st.warning("Biblioteca HPLC não disponível. A detecção de picos será ignorada.")
                                # Criar um objeto Chromatogram básico para o gráfico se a lib não existir,
                                # ou pelo menos ter um DataFrame para o gráfico
                                # A biblioteca HPLC requer a instalação para criar o objeto Chromatogram
                                # Então, se não está disponível, apenas usamos o pandas DataFrame
                                # e mostramos um gráfico básico.

                            # --- Mostrar Resultados ---
                            st.header("📈 Resultados da Análise")

                            # Gráfico do cromatograma
                            col_graph, col_info = st.columns([2, 1])

                            with col_graph:
                                st.subheader("Cromatograma")
                                fig, ax = plt.subplots(figsize=(12, 6))

                                # Usar o DataFrame carregado para plotar
                                if data_df is not None and time_col in data_df.columns and signal_col in data_df.columns:
                                    ax.plot(data_df[time_col], data_df[signal_col], 'b-', linewidth=1)

                                    # Se picos foram detectados, plotar os picos
                                    if not dados_picos.empty and 'rt' in dados_picos.columns and 'height' in dados_picos.columns:
                                         ax.plot(dados_picos['rt'], dados_picos['height'], 'ro', markersize=5, label='Picos Detetados')
                                         ax.vlines(dados_picos['rt'], [0], dados_picos['height'], color='red', linestyle=':', linewidth=0.8)
                                         ax.legend()


                                    ax.set_xlabel(f'{time_col} ({data_df[time_col].dtype})') # Mostrar tipo de dado
                                    ax.set_ylabel(f'{signal_col} ({data_df[signal_col].dtype})') # Mostrar tipo de dado
                                else:
                                     ax.set_xlabel("Tempo")
                                     ax.set_ylabel("Sinal")
                                     st.error("Não foi possível plotar o cromatograma. Verifique as colunas e os dados.")


                                ax.set_title('Cromatograma com Picos Detetados')
                                ax.grid(True, alpha=0.3)
                                st.pyplot(fig)
                                plt.close(fig) # Fecha a figura para liberar memória

                            with col_info:
                                st.subheader("Informações dos Picos")
                                if HPLC_LIB_AVAILABLE:
                                    if not dados_picos.empty:
                                        st.metric("Número de picos encontrados", len(dados_picos))

                                        # Mostrar tabela de picos
                                        st.dataframe(dados_picos)

                                        # Opção para download
                                        csv = dados_picos.to_csv(index=False)
                                        st.download_button(
                                            label="💾 Transferir dados dos picos (CSV)",
                                            data=csv,
                                            file_name="picos_detectados.csv",
                                            mime="text/csv",
                                            key="download_peaks_button"
                                        )

                                        # Estatísticas dos picos
                                        st.subheader("📈 Estatísticas dos Picos")
                                        if 'height' in dados_picos.columns and 'area' in dados_picos.columns:
                                            col_h1, col_h2, col_h3 = st.columns(3)
                                            with col_h1:
                                                st.metric("Altura média", f"{dados_picos['height'].mean():,.3f}")
                                            with col_h2:
                                                st.metric("Altura máxima", f"{dados_picos['height'].max():,.3f}")
                                            with col_h3:
                                                st.metric("Altura mínima", f"{dados_picos['height'].min():,.3f}")

                                            col_a1, col_a2, col_a3 = st.columns(3)
                                            with col_a1:
                                                st.metric("Área Total", f"{dados_picos['area'].sum():,.3f}")
                                            with col_a2:
                                                st.metric("Área média", f"{dados_picos['area'].mean():,.3f}")
                                            with col_a3:
                                                st.metric("Área máxima", f"{dados_picos['area'].max():,.3f}")

                                    else:
                                        st.warning("Nenhum pico foi detectado com os parâmetros atuais.")
                                else:
                                     st.info("Análise de picos requer a biblioteca HPLC.")


                        except Exception as e:
                            st.error(f"❌ Ocorreu um erro durante o processamento dos dados: {str(e)}")
                            st.exception(e)


        except Exception as e:
            st.error(f"❌ Erro ao ler o ficheiro com o delimitador '{delimiter}': {str(e)}")
            st.info("💡 Tente ajustar o delimitador ou verificar se o ficheiro é realmente CSV/DAT compatível.")
            st.exception(e)

    else:
        st.info("👆 Por favor, carregue um ficheiro CSV ou DAT para começar a análise.")

        # Exemplo de formato esperado
        st.header("📝 Formato Esperado do Ficheiro")
        st.markdown("""
        O ficheiro deve conter pelo menos duas colunas com dados numéricos:
        - Uma coluna para o **Tempo** (e.g., em minutos, segundos, milisegundos).
        - Uma coluna para o **Sinal** (e.g., absorbância, intensidade).

        As colunas devem estar separadas pelo delimitador escolhido (vírgula, ponto e vírgula, tabulação, espaço).

        **Exemplo de formato CSV (com vírgula como delimitador):**
        ```csv
        Tempo (min),Sinal (mAU)
        0.1,0.05
        0.2,0.12
        0.3,0.08
        0.4,0.55
        0.5,1.20
        0.6,0.80
        ...
        ```

        **Nota:** A aplicação irá converter a coluna de tempo para minutos para o processamento e análise.
        """)

if __name__ == "__main__":
    main()