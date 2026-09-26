#!/usr/bin/env python3
"""Render the briefing as a single HTML file with embedded figures."""
import base64
import html
from pathlib import Path
import re
from markdown_it import MarkdownIt
from presentation_panels import inject, CSS as panel_css

base=Path(__file__).resolve().parents[2]
source=base/'BASKETBALL_V2_PRESENTATION_REPORT.md'
body=inject(MarkdownIt('commonmark').enable('table').render(source.read_text()))
body=body.replace('<blockquote>\n<p><strong>쉬운 말로 세 줄 요약</strong>', '<blockquote class="easy-summary" aria-label="쉬운 말로 세 줄 요약">\n<p><strong>쉬운 말로 세 줄 요약</strong>')
def embed(match):
    path=base/match[1]
    encoded=base64.b64encode(path.read_bytes()).decode()
    return 'src="data:image/png;base64,'+encoded+'"'
body=re.sub(r'src="(analysis/basketball_v2/[^\"]+\.png)"',embed,body)
css='''
:root{color-scheme:light}body{font-family:"Noto Sans KR","Malgun Gothic",sans-serif;color:#182431;background:#f3f5f7;margin:0;line-height:1.75;font-size:15px}
main{max-width:1080px;margin:32px auto;padding:48px 60px;background:white;box-shadow:0 4px 28px #18243112}
h1{font-size:30px;line-height:1.35;color:#183d59}h2{margin-top:54px;border-bottom:2px solid #3e729c;padding-bottom:10px;color:#183d59}h3{margin-top:30px;font-size:19px}
p,li{word-break:keep-all;overflow-wrap:anywhere}table{width:100%;border-collapse:collapse;font-size:13px;margin:20px 0;line-height:1.6}th,td{border:1px solid #dce3e8;padding:9px 10px;text-align:left;vertical-align:top}th{background:#edf3f7}tbody tr:nth-child(even){background:#fafbfc}
img{max-width:100%;height:auto}blockquote{border-left:4px solid #3e729c;padding:10px 20px;background:#f1f6fa;margin:24px 0}code{font-family:monospace;font-size:.88em;background:#f0f3f5;padding:2px 4px;overflow-wrap:anywhere}pre{white-space:pre-wrap;background:#f0f3f5;padding:16px}a{color:#245d87}nav{background:#f1f6fa;padding:16px 24px;border-radius:4px}nav a{margin-right:18px;text-decoration:none;display:inline-block}.print-note{color:#617386;font-size:13px}
@media(max-width:760px){main{padding:24px 16px;margin:0}table{font-size:11px}th,td{padding:6px}h1{font-size:25px}}
@media print{@page{size:A4;margin:16mm}body{background:white;font-size:10pt}main{box-shadow:none;max-width:none;margin:0;padding:0}h1{font-size:22pt}h2{break-before:page;font-size:17pt;margin-top:0}h3{break-after:avoid}table{font-size:8pt}tr,img,blockquote{break-inside:avoid}thead{display:table-header-group}nav,.print-note{display:none}a{color:inherit;text-decoration:none}}
'''
css += panel_css
css += """
.easy-summary{background:#edf7f3;border-left:5px solid #008577;border-radius:0 10px 10px 0;padding:16px 22px;break-inside:avoid}
.easy-summary>p{margin:0;color:#006c60;font-size:17px}.easy-summary ol{padding-left:24px;margin:10px 0 0}.easy-summary li{margin:8px 0;font-size:16px;line-height:1.7}
@media print{.easy-summary{padding:10px 14px}.easy-summary li,.easy-summary>p{font-size:10pt}}
"""
labels=[]
index=0
def heading(match):
    global index
    index+=1
    labels.append((index,match[1]))
    return f'<h2 id="section-{index}">{match[1]}</h2>'
body=re.sub(r'<h2>(.*?)</h2>',heading,body)
body=body.replace('<h3>2.1 문헌으로 이해하는 단일 구간과 다중 구간 검색</h3>', '<h3 id="literature">2.1 문헌으로 이해하는 단일 구간과 다중 구간 검색</h3>')
nav='<nav>'+''.join(f'<a href="#section-{i}">{label}</a>' for i,label in labels)+'<a href="#literature">다중 구간 문헌 설명</a></nav>'
page='<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>농구 영상 그라운딩 실험 보고서</title><style>'+css+'</style></head><body><main><p class="print-note">발표용 보고서 · 그림이 포함된 단일 HTML 파일 · 브라우저 인쇄 → PDF 저장 가능</p>'+nav+body+'</main></body></html>'
target=source.with_suffix('.html');target.write_text(page)
assert page.count('<table>')>=10 and page.count('src="data:image/png;base64,')==5
print(target)
