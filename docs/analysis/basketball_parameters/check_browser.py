import json
from pathlib import Path
from playwright.sync_api import sync_playwright
root=Path(__file__).resolve().parents[2]
path=root/'BASKETBALL_EXPERIMENT2_PRESENTATION_REPORT.html'
with sync_playwright() as p:
 b=p.chromium.launch(args=['--no-sandbox'])
 page=b.new_page(viewport={'width':1440,'height':1000})
 errors=[]
 page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto(path.as_uri());page.wait_for_load_state('load')
 assert page.locator('img').count()==8
 assert page.locator('img').evaluate_all('(xs)=>xs.every(x=>x.complete && x.naturalWidth>0)')
 assert page.locator('#timeline rect[role=button]').count()==34
 page.select_option('#model','timelens-8b');page.select_option('#condition','C')
 assert '반환 30개' in page.locator('#metrics').inner_text()
 page.select_option('#query','football')
 assert page.locator('#timeline rect[role=button]').count()==0
 assert page.locator('#raw').inner_text().strip()=='[]'
 page.select_option('#query','bench')
 page.locator('#timeline rect[role=button]').first.click()
 assert '정답 여부 미검수' in page.locator('#selection').inner_text()
 page.locator('.explorer').screenshot(path='/tmp/basketball_exp2_explorer.png')
 page.screenshot(path='/tmp/basketball_exp2_desktop.png')
 links=page.locator('a').evaluate_all('(xs)=>xs.map(x=>x.getAttribute("href"))')
 for link in links:
  if link.startswith('#'):assert page.locator(link).count()==1,link
  elif not '://' in link:assert (root/link).exists(),link
 page.set_viewport_size({'width':390,'height':844})
 dims=page.evaluate('({width:innerWidth,scroll:document.documentElement.scrollWidth})')
 page.screenshot(path='/tmp/basketball_exp2_mobile.png')
 assert dims['scroll']<=dims['width'],dims
 assert not errors,errors
 result={'embedded_figures':8,'desktop_and_mobile':True,'javascript_errors':errors,'model_query_condition_selectors':True,'interval_selection':True,'relative_links_valid':True,'mobile_dimensions':dims}
 (root/'analysis/basketball_parameters/browser_validation.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result))
 b.close()
