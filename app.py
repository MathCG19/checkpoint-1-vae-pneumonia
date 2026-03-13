import os
import json
import io
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

# ==============================================
# App Streamlit para VAE PneumoniaMNIST
# ==============================================
# Funcionalidades:
# - Triagem de pneumonia baseada no erro de reconstrução
# - Geração de novas imagens de raio-X
# - Upload e reconstrução de imagens
# ==============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
WEIGHTS_PATH = os.path.join(MODELS_DIR, 'vae_pneumonia.weights.h5')
CONFIG_PATH = os.path.join(MODELS_DIR, 'config.json')


class Sampling(tf.keras.layers.Layer):
    def call(self, inputs, **kwargs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_encoder(latent_dim: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var')(x)
    z = Sampling()([z_mean, z_log_var])
    return tf.keras.Model(inputs, [z_mean, z_log_var, z], name='encoder')


def build_decoder(latent_dim: int) -> tf.keras.Model:
    latent_inputs = tf.keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(7 * 7 * 64, activation='relu')(latent_inputs)
    x = tf.keras.layers.Reshape((7, 7, 64))(x)
    x = tf.keras.layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    outputs = tf.keras.layers.Conv2DTranspose(1, kernel_size=3, padding='same', activation='sigmoid')(x)
    return tf.keras.Model(latent_inputs, outputs, name='decoder')


class VAE(tf.keras.Model):
    def __init__(self, encoder: tf.keras.Model, decoder: tf.keras.Model, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs, training=False):
        z_mean, z_log_var, z = self.encoder(inputs, training=training)
        reconstruction = self.decoder(z, training=training)
        return reconstruction

    def encode(self, inputs, training=False):
        return self.encoder(inputs, training=training)

    def decode(self, z, training=False):
        return self.decoder(z, training=training)


@st.cache_resource
def load_model():
    """Carrega o VAE uma vez; resultado em cache evita recarregar a cada rerun."""
    if not os.path.exists(CONFIG_PATH) or not os.path.exists(WEIGHTS_PATH):
        return None, 'Pesos ou configuração não encontrados. Treine o modelo executando train_vae.py.'
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)
    latent_dim = int(config.get('latent_dim', 16))
    encoder = build_encoder(latent_dim)
    decoder = build_decoder(latent_dim)
    vae = VAE(encoder, decoder)
    # Construir o modelo chamando uma passagem dummy antes de carregar pesos
    dummy = tf.zeros((1, 28, 28, 1))
    _ = vae(dummy, training=False)
    vae.load_weights(WEIGHTS_PATH)
    return vae, None


def preprocess_image(image: Image.Image) -> np.ndarray:
    # Converter para grayscale e 28x28
    if image.mode != 'L':
        image = image.convert('L')
    if image.size != (28, 28):
        image = image.resize((28, 28))
    arr = np.array(image).astype('float32')
    if arr.max() > 1.0:
        arr = arr / 255.0
    arr = np.expand_dims(arr, axis=-1)  # (28,28,1)
    arr = np.expand_dims(arr, axis=0)   # (1,28,28,1)
    return arr


def compute_reconstruction_error(x: np.ndarray, x_recon: np.ndarray) -> float:
    # Erro MSE por imagem
    return float(np.mean((x - x_recon) ** 2))


def classify_pneumonia(reconstruction_error: float) -> tuple:
    """
    Classifica se há possível pneumonia baseado no erro de reconstrução.
    Retorna: (classificação, descrição, cor, nível_confiança).
    Erro alto = possível pneumonia (imagem fora do padrão normal aprendido).
    """
    # Thresholds baseados em experiência com o dataset (ajustar conforme necessário)
    if reconstruction_error < 0.01:
        return "NORMAL", "Baixo risco de pneumonia", "green", "alta"
    elif reconstruction_error < 0.02:
        return "BORDERLINE", "Risco moderado - recomenda-se avaliação médica", "orange", "média"
    else:
        return "POSSÍVEL PNEUMONIA", "Alto risco - urgente avaliação médica", "red", "baixa"


def generate_new_images(vae: VAE, num_images: int = 4) -> np.ndarray:
    """Gera novas imagens de raio-X usando o VAE treinado."""
    latent_dim = vae.encoder.output_shape[0][-1]  # Pega a dimensão do z_mean
    
    # Amostrar do espaço latente normal padrão
    z_samples = np.random.normal(0, 1, (num_images, latent_dim))
    
    # Decodificar para gerar imagens
    generated_images = vae.decode(z_samples, training=False).numpy()
    
    return generated_images


st.set_page_config(page_title='VAE PneumoniaMNIST - Triagem e Geração', layout='wide')
st.title('VAE PneumoniaMNIST - Triagem de Pneumonia e Geração de Imagens')

# Estado da triagem: só exibir resultado após clicar em "Analisar"
if 'analysis_ran' not in st.session_state:
    st.session_state.analysis_ran = False
if 'triagem_result' not in st.session_state:
    st.session_state.triagem_result = None
if 'feedback_history' not in st.session_state:
    st.session_state.feedback_history = []  # lista de dicts: timestamp, classification, confidence, image_bytes, feedback
if 'feedback_given_for_current' not in st.session_state:
    st.session_state.feedback_given_for_current = False
if 'execution_history' not in st.session_state:
    st.session_state.execution_history = []  # histórico operacional: cada execução de análise (timestamp, image_bytes, mse, classification, confidence)


def reset_triagem_result():
    """Callback: ao trocar o arquivo enviado, reseta análise, resultado e feedback da análise atual."""
    st.session_state.analysis_ran = False
    st.session_state.triagem_result = None
    st.session_state.feedback_given_for_current = False


def clear_generated_images():
    """Callback: ao alterar parâmetros da geração, limpa as imagens geradas para evitar inconsistência."""
    if 'generated_images' in st.session_state:
        del st.session_state.generated_images


# Sidebar: painel de controle global (status do modelo, resumo, orientação)
with st.sidebar:
    st.header('⚙️ Painel de controle')
    st.markdown("---")
    st.subheader('Modelo VAE')
    vae, err = load_model()
    if err:
        st.error(err)
        st.stop()
    else:
        st.success('✅ Modelo carregado')
        st.caption(f"Dimensão latente: {vae.encoder.output_shape[0][-1]}")
    st.markdown("---")
    st.subheader('Monitoramento')
    n_exec = len(st.session_state.execution_history)
    n_feedback = len(st.session_state.feedback_history)
    st.metric("Total de análises", n_exec)
    st.metric("Feedbacks registrados", n_feedback)
    st.caption("Histórico operacional e de feedback na aba **Triagem**.")
    st.markdown("---")
    st.subheader('Navegação')
    st.caption("• **Triagem:** envie imagem e analise. Históricos abaixo.")
    st.caption("• **Gerar:** defina a quantidade e clique em Gerar.")
    st.caption("• **Sobre:** arquitetura e limites do modelo.")

# Área principal: abas por contexto (triagem, geração, documentação)
st.markdown("---")
tab1, tab2, tab3 = st.tabs(["🔍 Triagem de Pneumonia", "🎨 Gerar Novas Imagens", "📊 Sobre o Modelo"])

with tab1:
    st.header("Triagem de Pneumonia via VAE")
    st.caption("Envie uma imagem de raio-X para análise de triagem baseada em erro de reconstrução do VAE.")
    st.markdown("---")
    
    st.subheader("Entrada")
    st.markdown("**Como funciona:** O VAE foi treinado em imagens normais. Imagens com pneumonia tendem a ter maior erro de reconstrução.")
    uploaded = st.file_uploader(
        'Envie uma imagem de raio-X para análise (PNG/JPG)',
        type=['png', 'jpg', 'jpeg'],
        key='upload_triagem',
        on_change=reset_triagem_result
    )
    
    # Empty state: sem imagem enviada
    if uploaded is None:
        st.info("👆 **Envie uma imagem** de raio-X (PNG ou JPG) acima e, em seguida, clique em **Analisar** para ver o resultado da triagem.")
        st.markdown("---")
        if st.session_state.analysis_ran or st.session_state.triagem_result is not None:
            st.session_state.analysis_ran = False
            st.session_state.triagem_result = None
    else:
        # Imagem enviada: mostrar preview e botão Analisar
        image = Image.open(io.BytesIO(uploaded.getvalue()))
        x = preprocess_image(image)
        
        if not st.session_state.analysis_ran:
            # Estado antes da análise: orientar e oferecer o botão
            st.info("✅ Imagem carregada. Clique em **Analisar** abaixo para executar a triagem.")
            col_preview, _ = st.columns([1, 2])
            with col_preview:
                st.image(x[0].squeeze(), clamp=True, use_column_width=True, caption="Pré-visualização")
            if st.button("🔍 Analisar", type="primary", key="btn_analisar"):
                with st.status("Iniciando pipeline de triagem...", expanded=True) as status:
                    progress = st.progress(0.0)
                    # Etapa 1: Carregando imagem
                    status.update(label="Carregando imagem...", state="running")
                    progress.progress(1 / 3)
                    # (imagem já pré-processada em x)
                    # Etapa 2: Reconstruindo
                    status.update(label="Reconstruindo...", state="running")
                    progress.progress(2 / 3)
                    recon = vae(x, training=False).numpy()
                    # Etapa 3: Classificando
                    status.update(label="Classificando...", state="running")
                    progress.progress(1.0)
                    mse = compute_reconstruction_error(x, recon)
                    classification, description, color, confidence_level = classify_pneumonia(mse)
                    status.update(label="Análise concluída", state="complete")
                confidence_value = min(1.0, max(0.0, 1.0 - mse))
                img_arr = (x[0].squeeze() * 255).astype(np.uint8)
                img_pil = Image.fromarray(img_arr, mode='L')
                buf = io.BytesIO()
                img_pil.save(buf, format='PNG')
                st.session_state.triagem_result = {
                    'x': x, 'recon': recon, 'mse': mse,
                    'classification': classification, 'description': description, 'color': color,
                    'confidence_level': confidence_level
                }
                st.session_state.execution_history.append({
                    'timestamp': datetime.now(),
                    'image_bytes': buf.getvalue(),
                    'mse': mse,
                    'classification': classification,
                    'confidence': confidence_value
                })
                st.session_state.analysis_ran = True
                st.session_state.feedback_given_for_current = False
                st.rerun()
        else:
            # Análise já executada: exibir resultado persistido
            res = st.session_state.triagem_result
            if res is not None:
                st.subheader("Resultado")
                col1, col2 = st.columns(2)
                with col1:
                    st.caption("Imagem original")
                    st.image(res['x'][0].squeeze(), clamp=True, use_column_width=True)
                with col2:
                    st.caption("Reconstrução VAE")
                    st.image(res['recon'][0].squeeze(), clamp=True, use_column_width=True)
                
                st.markdown("---")
                st.caption("Métricas da triagem")
                
                mse = res['mse']
                confidence_value = min(1.0, max(0.0, 1.0 - mse))  # proporcional à confiança
                confidence_pct = confidence_value * 100
                confidence_level = res.get('confidence_level', 'média')
                confidence_label = confidence_level.capitalize()  # Alta / Média / Baixa
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Erro de Reconstrução", f"{mse:.6f}")
                with col2:
                    st.metric("Classificação", res['classification'])
                with col3:
                    st.metric("Confiança", f"{confidence_pct:.1f}%")
                
                st.markdown("**Nível de confiança da triagem:**")
                st.progress(confidence_value)
                st.caption(f"**{confidence_label} confiança** — Esta é uma **estimativa** do modelo, não um diagnóstico médico.")
                
                # Alerta semântico conforme classificação (st.success / st.warning / st.error)
                if res['classification'] == "NORMAL":
                    st.success(f"**{res['classification']}** — {res['description']}")
                elif res['classification'] == "BORDERLINE":
                    st.warning(f"**{res['classification']}** — {res['description']}")
                else:
                    st.error(f"**{res['classification']}** — {res['description']}")
                
                # Orientação comportamental quando confiança baixa
                if confidence_level == "baixa":
                    st.info("👨‍⚕️ **Recomendamos revisão por um profissional de saúde** para conclusão diagnóstica.")
                
                st.caption("⚠️ **Importante:** Este é apenas um auxiliar de triagem. O resultado é uma **estimativa** e não substitui avaliação médica. Sempre consulte um médico para diagnóstico definitivo.")
                
                st.markdown("---")
                st.subheader("Feedback (Human-in-the-loop)")
                st.caption("A classificação estava correta?")
                col_acertou, col_errou, _ = st.columns([1, 1, 2])
                if not st.session_state.feedback_given_for_current:
                    with col_acertou:
                        if st.button("Acertou ✅", key="btn_acertou", type="primary"):
                            img_arr = (res['x'][0].squeeze() * 255).astype(np.uint8)
                            img_pil = Image.fromarray(img_arr, mode='L')
                            buf = io.BytesIO()
                            img_pil.save(buf, format='PNG')
                            st.session_state.feedback_history.append({
                                'timestamp': datetime.now(),
                                'classification': res['classification'],
                                'confidence': confidence_value,
                                'image_bytes': buf.getvalue(),
                                'feedback': 'Acertou'
                            })
                            st.session_state.feedback_given_for_current = True
                            st.toast("Obrigado! Registramos que a classificação estava correta.", icon="✅")
                            st.rerun()
                    with col_errou:
                        if st.button("Errou ❌", key="btn_errou"):
                            img_arr = (res['x'][0].squeeze() * 255).astype(np.uint8)
                            img_pil = Image.fromarray(img_arr, mode='L')
                            buf = io.BytesIO()
                            img_pil.save(buf, format='PNG')
                            st.session_state.feedback_history.append({
                                'timestamp': datetime.now(),
                                'classification': res['classification'],
                                'confidence': confidence_value,
                                'image_bytes': buf.getvalue(),
                                'feedback': 'Errou'
                            })
                            st.session_state.feedback_given_for_current = True
                            st.toast("Obrigado! Registramos que a classificação estava incorreta.", icon="❌")
                            st.rerun()
                else:
                    st.caption("Feedback registrado para esta análise.")
    
    # Históricos visíveis sempre que houver dados (estado controla UI; sem comportamento inconsistente no re-run)
    # -------------------------------------------------------------------------
    # Histórico operacional (execuções) — separado do histórico de feedback
    # -------------------------------------------------------------------------
    if st.session_state.execution_history:
        st.markdown("---")
        st.subheader("📊 Histórico operacional (execuções)")
        exec_hist = st.session_state.execution_history
        df_exec = pd.DataFrame([
            {
                'Imagem': e['image_bytes'],
                'MSE': round(e['mse'], 6),
                'Classificação': e['classification'],
                'Confiança': e['confidence'],
                'Data/Hora': e['timestamp'].strftime('%d/%m/%Y %H:%M')
            }
            for e in exec_hist
        ])
        st.dataframe(
            df_exec,
            column_config={
                'Imagem': st.column_config.ImageColumn('Imagem', width='small'),
                'MSE': st.column_config.NumberColumn('MSE', format='%.6f', width='small'),
                'Classificação': st.column_config.TextColumn('Classificação', width='medium'),
                'Confiança': st.column_config.ProgressColumn('Confiança', format='percent', min_value=0.0, max_value=1.0, width='medium'),
                'Data/Hora': st.column_config.TextColumn('Data/Hora', width='medium')
            },
            hide_index=True,
            use_container_width=True
        )
        total_exec = len(exec_hist)
        avg_conf = sum(e['confidence'] for e in exec_hist) / total_exec
        count_normal = sum(1 for e in exec_hist if e['classification'] == 'NORMAL')
        count_border = sum(1 for e in exec_hist if e['classification'] == 'BORDERLINE')
        count_pneum = sum(1 for e in exec_hist if e['classification'] == 'POSSÍVEL PNEUMONIA')
        st.markdown("**Métricas agregadas**")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Total de análises", total_exec)
        with c2:
            st.metric("Média confiança", f"{avg_conf * 100:.1f}%")
        with c3:
            st.metric("% NORMAL", f"{count_normal / total_exec * 100:.0f}%")
        with c4:
            st.metric("% BORDERLINE", f"{count_border / total_exec * 100:.0f}%")
        with c5:
            st.metric("% POSSÍVEL PNEUMONIA", f"{count_pneum / total_exec * 100:.0f}%")
        low_conf_count = sum(1 for e in exec_hist if e['confidence'] < 0.5)
        if total_exec >= 3 and (low_conf_count / total_exec > 0.5 or count_pneum / total_exec > 0.5):
            st.warning("⚠️ **Possível degradação:** muitas análises com confiança baixa ou classificação POSSÍVEL PNEUMONIA. Considere revisar o modelo ou a qualidade dos dados.")
    
    # -------------------------------------------------------------------------
    # Histórico de feedback (avaliação humana) — separado do operacional
    # -------------------------------------------------------------------------
    if st.session_state.feedback_history:
        st.markdown("---")
        st.subheader("📋 Histórico de feedback (avaliação humana)")
        hist = st.session_state.feedback_history
        df_feedback = pd.DataFrame([
            {
                'Data/Hora': e['timestamp'].strftime('%d/%m/%Y %H:%M'),
                'Classificação': e['classification'],
                'Confiança': e['confidence'],
                'Imagem': e['image_bytes'],
                'Feedback': e['feedback']
            }
            for e in hist
        ])
        st.dataframe(
            df_feedback,
            column_config={
                'Data/Hora': st.column_config.TextColumn('Data/Hora', width='medium'),
                'Classificação': st.column_config.TextColumn('Classificação', width='medium'),
                'Confiança': st.column_config.ProgressColumn('Confiança', format='percent', min_value=0.0, max_value=1.0, width='medium'),
                'Imagem': st.column_config.ImageColumn('Imagem', width='small'),
                'Feedback': st.column_config.TextColumn('Feedback', width='small')
            },
            hide_index=True,
            use_container_width=True
        )
        erros = sum(1 for e in hist if e['feedback'] == 'Errou')
        total = len(hist)
        if total >= 3 and erros / total > 0.5:
            st.warning("⚠️ **Possível degradação do modelo:** muitos feedbacks indicando classificação incorreta. Considere revisar o modelo ou os dados de treinamento.")
        st.markdown("**Evolução da confiança nas interações**")
        df_chart = pd.DataFrame({'Interação': range(1, total + 1), 'Confiança': [e['confidence'] for e in hist]})
        st.line_chart(df_chart.set_index('Interação'), y='Confiança', height=250)

with tab2:
    st.header("🎨 Geração de Novas Imagens de Raio-X")
    st.markdown("""
    Gere novas imagens sintéticas de raio-X usando o espaço latente aprendido pelo VAE.
    """)
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        num_images = st.slider(
            "Número de imagens a gerar",
            min_value=1,
            max_value=8,
            value=4,
            key="num_images_slider",
            on_change=clear_generated_images,
            help="Alterar este valor limpa as imagens já geradas."
        )
        if st.button("🔄 Gerar Novas Imagens", type="primary", key="btn_gerar"):
            with st.spinner("Gerando imagens..."):
                generated = generate_new_images(vae, num_images)
                st.session_state.generated_images = generated
    
    with col2:
        st.markdown("**Controles**")
        st.caption("Ajuste o número de imagens e clique em **Gerar**. Alterar o valor reseta a saída.")
        st.caption("Imagens amostradas do espaço latente normal.")
    
    # Empty state: ainda não gerou imagens
    if 'generated_images' not in st.session_state:
        st.info("👆 Ajuste o **número de imagens** acima e clique em **Gerar Novas Imagens** para criar imagens sintéticas de raio-X.")
    else:
        n_show = len(st.session_state.generated_images)
        st.subheader("Imagens Geradas")
        cols = st.columns(n_show)
        for i, col in enumerate(cols):
            with col:
                st.image(st.session_state.generated_images[i].squeeze(),
                            clamp=True,
                            caption=f"Imagem {i+1}",
                            use_column_width=True)
        st.markdown("---")
        if st.button("💾 Salvar Imagens", key="btn_salvar_geradas"):
            images = []
            for i in range(n_show):
                img_array = (st.session_state.generated_images[i].squeeze() * 255).astype(np.uint8)
                img = Image.fromarray(img_array, mode='L')
                images.append(img)
            st.success("Imagens geradas. Use print screen ou salve individualmente.")

with tab3:
    st.header("Sobre o Modelo VAE")
    st.caption("Arquitetura, funcionamento da triagem e limitações. Para analisar imagens ou gerar novas, use as abas **Triagem** e **Gerar**.")
    st.markdown("---")
    
    st.subheader("Arquitetura do Modelo")
    st.markdown("""
    **Encoder:**
    - Conv2D(32) → Conv2D(64) → Flatten → Dense(128) → Latent Space
    
    **Decoder:**
    - Dense(7×7×64) → Reshape → Conv2DTranspose(64) → Conv2DTranspose(32) → Output
    
    **Dimensão Latente:** 16 variáveis
    """)
    st.markdown("---")
    st.subheader("Como Funciona a Triagem")
    st.markdown("""
    
    1. **Imagens Normais:** Baixo erro de reconstrução (padrão bem aprendido)
    2. **Imagens com Pneumonia:** Alto erro de reconstrução (padrão diferente)
    3. **Thresholds:** 
       - < 0.01: Normal
       - 0.01-0.02: Borderline  
       - > 0.02: Possível pneumonia
    """)
    st.markdown("---")
    st.subheader("Limitações")
    st.markdown("""
    - Treinado apenas em PneumoniaMNIST
    - Não substitui diagnóstico médico
    - Sensibilidade depende da qualidade da imagem
    """)
    st.markdown("---")
    if vae:
        st.subheader("Estatísticas do Modelo")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Parâmetros Encoder", f"{vae.encoder.count_params():,}")
            st.metric("Parâmetros Decoder", f"{vae.decoder.count_params():,}")
        with col2:
            st.metric("Total Parâmetros", f"{vae.count_params():,}")
            st.metric("Dimensão Latente", vae.encoder.output_shape[0][-1])

# Footer
st.markdown("---")
st.caption("""
🔬 **Modelo VAE para Triagem de Pneumonia** | 
Desenvolvido com TensorFlow e Streamlit | 
Sempre consulte um médico para diagnóstico definitivo.
""") 