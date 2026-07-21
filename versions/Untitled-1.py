if use_gemini == "1":

    # 誤字・表記統一・自然な校正（Gemini）
    # 本文分離方式

    import concurrent.futures
    import time
    import re


    def build_gemini_prompt(text):

        return f"""
以下の字幕データを校正してください。

【目的】
・誤字脱字の修正
・明らかな誤変換の修正
・固有名詞の表記統一
・漢字、かな表記の統一
・自然な日本語への軽微な修正


【禁止事項】
・意味を変更しない
・文章を要約しない
・文章量を減らさない
・フィラー削除をしない
・字幕番号を変更しない
・字幕ブロックを分割しない
・字幕ブロックを統合しない
・新しい字幕番号を追加しない
・既存の改行位置を変更しない


【最重要】

入力された字幕番号ごとの構造を完全に維持してください。

番号は字幕ブロックを識別するIDです。
各番号は必ず同じ番号のまま返してください。

1つの字幕番号は、必ず1つの字幕ブロックとして返してください。

各字幕ブロック内の文章量、改行数、改行位置は変更禁止です。

修正できるのは文字単位の修正のみです。
字幕レイアウトや文章構造には一切変更を加えないでください。


【出力形式】

入力と同じ番号形式で出力してください。

番号付き字幕ブロックのみ出力してください。
説明文やコメントは追加しないでください。


【入力】

{text}
"""

    def extract_text_for_gemini(chunks):

        """
        Geminiへ送る本文だけを作成
        タイムコードは送らない
        """

        lines = []

        for i, c in enumerate(chunks, start=1):

            lines.append(
                f"{i}\n{c['text']}"
            )

        return "\n\n".join(lines)



    def parse_gemini_result(text):

        """
        Gemini結果から番号ごとの本文を取得
        """

        blocks = re.split(
            r"\n\s*\n",
            text.strip()
        )

        result = {}

        for block in blocks:

            lines = block.strip().split("\n")

            if len(lines) >= 2:

                try:
                    num = int(lines[0].strip())

                    body = "\n".join(
                        lines[1:]
                    )

                    result[num] = body

                except:

                    pass

        return result



    def generate_with_retry(
            prompt,
            max_retries=5,
            timeout_sec=120):


        wait = 10


        for attempt in range(1, max_retries + 1):

            try:

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:

                    future = executor.submit(
                        model.generate_content,
                        prompt
                    )

                    response = future.result(
                        timeout=timeout_sec
                    )


                if not response.text:

                    raise Exception(
                        "empty response"
                    )


                return response.text



            except Exception as e:


                error = str(e)

                print(
                    f"Geminiエラー {attempt}/{max_retries}"
                )

                print(error)



                # 429は即停止

                if (
                    "429" in error
                    or "TooManyRequests" in error
                    or "quota" in error.lower()
                ):

                    print("")
                    print("==============================")
                    print("⚠️ Gemini API無料枠上限です")
                    print("処理を停止します")
                    print("==============================")

                    raise RuntimeError(
                        "Gemini quota exceeded"
                    )



                # 503 / Timeoutのみリトライ

                if (
                    "503" in error
                    or isinstance(e, TimeoutError)
                ):

                    if attempt < max_retries:

                        print(
                            f"{wait}秒待機して再試行"
                        )

                        time.sleep(wait)

                        wait = min(
                            wait * 2,
                            300
                        )

                        continue



                raise e




    # =========================
    # メイン処理
    # =========================


    if "gemini_segments" not in locals():

        raise NameError(
            "gemini_segments がありません。前の処理を先に実行してください。"
        )


    batch_size = 100


    batches = [
        gemini_segments[i:i + batch_size]

        for i in range(
            0,
            len(gemini_segments),
            batch_size
        )
    ]


    print(
        f"{len(batches)}個のバッチに分割しました"
    )



    new_segments = []


    for idx, batch in enumerate(
            batches,
            start=1):


        print(
            f"バッチ {idx}/{len(batches)} 処理中..."
        )


        # 本文だけ抽出

        gemini_text = extract_text_for_gemini(
            batch
        )


        prompt = build_gemini_prompt(
            gemini_text
        )


        result = generate_with_retry(
            prompt
        )


        # 番号別に戻す

        corrected = parse_gemini_result(
            result
        )



        for i, c in enumerate(batch, start=1):

            new_chunk = c.copy()


            if i in corrected:

                new_chunk["text"] = corrected[i]


            else:

                print(
                    f"⚠️ {i}番の戻し失敗。元文章維持"
                )


            new_segments.append(
                new_chunk
            )



        # 連続送信防止

        if idx < len(batches):

            time.sleep(3)



    gemini_segments = new_segments


    current_srt = chunks_to_srt_text(
        gemini_segments
    )


    print("")
    print("=== Gemini校正完了 ===")
    print(current_srt)

else:

    print("Gemini校正をスキップしました。")

    current_srt = chunks_to_srt_text(
        gemini_segments
    )