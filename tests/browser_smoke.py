import json
import os
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    options = {'headless': True}
    if os.environ.get('MTCP_CHROMIUM'):
        options['executable_path'] = os.environ['MTCP_CHROMIUM']
    browser = p.chromium.launch(**options)
    page = browser.new_page(viewport={'width':1440,'height':1100}, device_scale_factor=1)
    errors=[]
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.goto(os.environ.get('MTCP_URL', 'http://127.0.0.1:8087'))
    page.wait_for_function("() => /^[1-8] volunteers connected$/.test(document.querySelector('#connection').textContent)")
    page.screenshot(path='/tmp/mtcp-desktop-empty.png', full_page=True)
    page.select_option('#scenario','corrupt')
    page.fill('#steps','8')
    page.click('#start')
    page.wait_for_function("() => document.querySelector('#run-status').textContent === 'running'", timeout=10000)
    page.wait_for_function("() => document.querySelector('#run-status').textContent === 'completed'", timeout=25000)
    assert page.locator('#rejected-count').inner_text() == '1 / 1'
    page.locator('[data-task]').first.click()
    assert page.locator('#task-detail').is_visible()
    assert 'artifact_digest' in page.locator('#detail-json').inner_text()
    page.click('#close-detail')
    page.click('#next-page')
    assert page.locator('#page-label').inner_text().startswith('2 /')
    page.click('#previous-page')
    page.screenshot(path='/tmp/mtcp-desktop-complete.png', full_page=True)
    page.select_option('#scenario','honest')
    page.fill('#steps','4')
    page.click('#start')
    page.wait_for_function("() => document.querySelector('#run-status').textContent === 'running'", timeout=10000)
    page.wait_for_function("() => document.querySelector('#run-status').textContent === 'completed'", timeout=15000)
    page.locator('[data-run]').nth(1).click()
    page.wait_for_function("() => document.querySelector('#run-status').textContent.includes('historical')")
    assert page.locator('#rejected-count').inner_text() == '1 / 1'
    page.set_viewport_size({'width':390,'height':844})
    page.screenshot(path='/tmp/mtcp-mobile.png', full_page=True)
    overflow=page.evaluate('document.documentElement.scrollWidth > window.innerWidth')
    assert not overflow, 'Mobile horizontal overflow'
    page.click('#live-view')
    assert page.locator('#rejected-count').inner_text() == '0 / 0'
    assert not page.locator('#live-view').is_visible()
    assert not errors, errors
    print(json.dumps({'browser_errors':errors, 'mobile_overflow':overflow,'checks':['corruption recovery','task inspector','pagination','run history','mobile layout']}))
    browser.close()
