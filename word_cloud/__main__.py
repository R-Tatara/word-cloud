#!/usr/bin/env python
# -*- cording: utf-8 -*-
import pandas as pd
from matplotlib import pyplot as plt
from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud


def generate_wordcloud(df_text):
    tokenizer = Tokenizer()

    # TF-IDFによるベクトル化
    nouns = df_text.apply(lambda x: extract_nouns(x, tokenizer))
    vectorizer = TfidfVectorizer(stop_words=['こと', 'ため', 'よう', 'もの', 'これ', 'それ', 'どこ', 'そこ', 'たい', 'ほか', 'さっき', 'びと'])
    X = vectorizer.fit_transform(nouns)
    df_vectorized = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

    # 上位のキーワードを抽出
    top_keywords = df_vectorized.sum().nlargest(200)

    # ワードクラウドの作成
    wordcloud = WordCloud(
        font_path='/System/Library/Fonts/ヒラギノ丸ゴ ProN W4.ttc',
        width=800,
        height=400,
        background_color='white',
        min_font_size=8,
        max_font_size=128,
        colormap='cividis'
    ).generate_from_frequencies(top_keywords)

    return wordcloud


def extract_nouns(text, tokenizer):
    tokens = tokenizer.tokenize(text)
    nouns = [token.base_form for token in tokens if token.part_of_speech.startswith('名詞')]
    return ' '.join(nouns)


def main():
    data = {
        'text': [
            'AI技術を活用した新型自動運転車、商業化に向けた実証実験開始',
            '量子コンピュータの商業利用が現実に、主要企業が相次ぎ発表',
            '5G通信インフラの整備が加速、全国主要都市でサービス開始',
            '次世代半導体技術の開発競争が激化、各国で研究開発が進展',
            'バイオテクノロジー企業、遺伝子編集技術の新たな応用例を発表',
            '再生可能エネルギーの効率化を目指す新型ソーラーパネルが登場',
            'スマートシティ構想が進展、AIを活用した都市管理システムが導入',
            '自動化技術の進化により、製造業の生産性が大幅に向上',
            'ドローン技術の進化により、物流業界での活用が拡大',
            '次世代バッテリー技術の開発が進み、電気自動車の航続距離が延長',
            'AIによる医療診断システムが導入され、診断精度が向上',
            'ロボット技術の進化により、高齢者介護の現場での活用が進む',
            'ブロックチェーン技術の応用が広がり、金融業界での利用が増加',
            '量子通信技術の商業化が進み、セキュリティ分野での利用が期待',
            '次世代インターネット技術の研究が進展、通信速度の大幅な向上が見込まれる',
            'AIを活用した教育プログラムが導入され、学習効果の向上が期待',
            '自動運転技術の進化により、交通事故の減少が期待される',
            '再生可能エネルギーの普及により、電力供給の安定性が向上',
            '次世代ロボット技術の開発が進み、家庭用ロボットの実用化が近づく',
            'AIによる気象予測システムが導入され、災害対策の精度が向上',
            'スマートホーム技術の進化により、生活の質が向上',
            '次世代医療機器の開発が進み、治療法の選択肢が広がる',
            '自動化技術の導入により、農業生産性が向上',
            'AIによる音声認識技術が進化し、コミュニケーションの効率化が進む',
            '次世代航空機の開発が進み、空の移動がより快適に',
            'ロボット技術の進化により、建設現場での作業効率が向上',
            'AIによる金融分析システムが導入され、投資判断の精度が向上',
            '次世代バッテリー技術の商業化が進み、電気自動車の普及が加速',
            '再生可能エネルギーの効率化技術が開発され、コスト削減が実現',
            'AIによる農業支援システムが導入され、作物の品質向上が期待',
            '自動運転技術の進化により、物流業界での効率化が進む',
            '次世代通信技術の研究が進展し、通信インフラの整備が加速',
            'AIによる製造業の品質管理システムが導入され、不良品率が低下',
            'ロボット技術の進化により、医療現場での手術支援が進む',
            '次世代エネルギー源の研究が進展し、持続可能な社会の実現が期待',
            'AIによる交通管理システムが導入され、渋滞の解消が期待',
            '再生可能エネルギーの普及により、エネルギー自給率が向上',
            '次世代ロボット技術の商業化が進み、サービス業での活用が拡大',
            'AIによる環境モニタリングシステムが導入され、環境保護が強化',
            '自動化技術の導入により、物流業界での人手不足が解消',
            '次世代医療技術の開発が進み、治療法の選択肢が広がる',
            'AIによるエネルギー管理システムが導入され、エネルギー効率が向上',
            'ロボット技術の進化により、教育現場での支援が進む',
            '次世代通信技術の商業化が進み、通信速度の大幅な向上が期待',
            'AIによる製造業の工程管理システムが導入され、生産性が向上',
            '再生可能エネルギーの効率化技術が開発され、エネルギーコストが削減',
            '次世代バッテリー技術の研究が進展し、電気自動車の普及が加速',
            'AIによる医療データ解析システムが導入され、個別化医療が進展',
            'ロボット技術の進化により、災害救助活動が効率化',
            '次世代エネルギー技術の商業化が進み、持続可能な社会の実現が期待',
            'AIによる教育支援システムが導入され、学習効果が向上',
            '自動化技術の導入により、製造業のコスト削減が実現',
            '次世代通信インフラの整備が進み、情報社会の基盤が強化',
            'AIによる環境保護システムが導入され、生態系の保全が進む',
            'ロボット技術の進化により、介護現場での支援が進む',
            '次世代医療機器の商業化が進み、治療法の選択肢が広がる'
        ]
    }

    # データフレームを作成
    df = pd.DataFrame(data)

    # ワードクラウドの作成
    wordcloud = generate_wordcloud(df['text'])

    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=3.0)
    plt.show()


if __name__ == "__main__":
    main()
