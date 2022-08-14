import logging

# def df_plot(df):
#     colors = ["#0101DF", "#DF0101"]

#     sns.countplot('Class', data=df, palette=colors)
#     plt.title('Class Distribution \n (0: No Fraud || 1:Fraud', fontsize=14)
#     return

def get_logger(message: str):
    # 로그 생성
    logger = logging.getLogger("GAN_FS")

    # 로그의 출력 기준 설정
    logger.setLevel(logging.INFO)

    # log 출력 형식
    formatter = logging.Formatter('%(asctime)s[%(levelname)s] [%(name)s] : %(message)s')

    # log 출력
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # log를 파일에 출력
    file_handler = logging.FileHandler('my.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f'{message}')
    
    return