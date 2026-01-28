#!/usr/bin/env python3
"""Generate a small sample Chinese EPUB for testing the graded reader converter."""

from ebooklib import epub

SAMPLE_CHAPTERS = [
    {
        'title': '第一章：到达北京',
        'content': '''
        <h1>第一章：到达北京</h1>
        <p>小明从上海坐火车到北京。他第一次来到这个城市，感到非常兴奋。</p>
        <p>火车站很大，人也很多。小明拿着行李，慢慢走出了车站。</p>
        <p>外面的天气很冷。他穿上厚厚的外套，开始找出租车。</p>
        <p>"师傅，请问去北京大学怎么走？"小明问出租车司机。</p>
        <p>"没问题，上车吧。"司机笑着说。</p>
        ''',
    },
    {
        'title': '第二章：新的朋友',
        'content': '''
        <h1>第二章：新的朋友</h1>
        <p>小明到了北京大学。校园很漂亮，有很多树和花。</p>
        <p>他在宿舍里认识了一个新朋友，叫李华。李华是北京人，很热情。</p>
        <p>"你好！我叫李华，欢迎来北京！"李华高兴地说。</p>
        <p>"谢谢！我叫小明，从上海来的。请多多关照。"小明回答。</p>
        <p>他们一起去食堂吃晚饭。食堂的菜很好吃，价格也不贵。</p>
        <p>晚上，小明给家里打了一个电话，告诉妈妈他已经安全到达了。</p>
        ''',
    },
    {
        'title': '第三章：第一天上课',
        'content': '''
        <h1>第三章：第一天上课</h1>
        <p>第二天早上，小明很早就起床了。他有点紧张，因为今天是第一天上课。</p>
        <p>教室在第三栋楼的二楼。他走进教室，发现已经有很多同学坐在里面了。</p>
        <p>老师是一位年轻的女老师，姓王。她说话的速度不快，很容易听懂。</p>
        <p>"同学们好，欢迎来到中文课。我希望大家这个学期都能学到很多东西。"</p>
        <p>小明认真地听课，还做了很多笔记。他觉得这个学校的老师都很好。</p>
        ''',
    },
]


def create_sample_epub(output_path: str = 'sample_chinese.epub'):
    book = epub.EpubBook()
    book.set_identifier('sample-chinese-001')
    book.set_title('到达北京 - 示例中文书')
    book.set_language('zh')
    book.add_author('测试作者')

    style = epub.EpubItem(
        uid='style_default',
        file_name='style/default.css',
        media_type='text/css',
        content=b'body { font-family: serif; line-height: 1.6; } p { margin-bottom: 0.5em; }',
    )
    book.add_item(style)

    chapters = []
    for i, ch in enumerate(SAMPLE_CHAPTERS, 1):
        c = epub.EpubHtml(
            title=ch['title'], file_name=f'chap_{i:02d}.xhtml', lang='zh'
        )
        c.content = f'''<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
  <title>{ch["title"]}</title>
  <link rel="stylesheet" href="style/default.css" type="text/css"/>
</head>
<body>
{ch["content"]}
</body>
</html>'''.encode('utf-8')
        c.add_item(style)
        book.add_item(c)
        chapters.append(c)

    book.toc = [epub.Link(f'chap_{i:02d}.xhtml', ch['title'], f'ch{i}')
                for i, ch in enumerate(SAMPLE_CHAPTERS, 1)]
    book.spine = ['nav'] + chapters
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    epub.write_epub(output_path, book, {})
    print(f'Sample EPUB created: {output_path}')


if __name__ == '__main__':
    create_sample_epub()
