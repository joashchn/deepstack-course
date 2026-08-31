
(function(){
  'use strict';
  var root = document.documentElement;

  /* ---- 侧边栏 ---- */
  var sidebar = document.getElementById('sidebar');
  var mask = document.getElementById('sbMask');
  var menuBtn = document.getElementById('menuBtn');
  var sbClose = document.getElementById('sbClose');
  function openSb(){ sidebar.classList.add('on'); mask.classList.add('on'); }
  function closeSb(){ sidebar.classList.remove('on'); mask.classList.remove('on'); }
  if (menuBtn) menuBtn.addEventListener('click', openSb);
  if (mask) mask.addEventListener('click', closeSb);
  if (sbClose) sbClose.addEventListener('click', closeSb);

  /* 分组展开：默认展开当前模块，记住用户手动切换 */
  var nav = document.querySelector('.sb-nav');
  if (nav) {
    var savedOpen = {};
    try { savedOpen = JSON.parse(localStorage.getItem('aidoc-open') || '{}'); } catch(e){}
    Array.prototype.forEach.call(nav.querySelectorAll('.sb-group'), function(g){
      var key = g.dataset.mod;
      if (savedOpen[key] === true) g.classList.add('open');
      if (savedOpen[key] === false) g.classList.remove('open');
      g.querySelector('.sb-group-title').addEventListener('click', function(e){
        if (e.metaKey || e.ctrlKey || e.shiftKey) return;
        e.preventDefault();
        g.classList.toggle('open');
        savedOpen[key] = g.classList.contains('open');
        try { localStorage.setItem('aidoc-open', JSON.stringify(savedOpen)); } catch(err){}
      });
    });
    var cur = nav.querySelector('.sb-item.current');
    if (cur) {
      cur.scrollIntoView({block:'center'});
    }
  }

  /* ---- 代码复制 ---- */
  Array.prototype.forEach.call(document.querySelectorAll('.highlight'), function(box){
    var btn = document.createElement('button');
    btn.className = 'copy-btn';
    btn.textContent = '复制';
    btn.addEventListener('click', function(){
      var code = box.querySelector('pre');
      var text = code ? code.innerText : '';
      var done = function(){
        btn.textContent = '已复制';
        btn.classList.add('done');
        setTimeout(function(){ btn.textContent = '复制'; btn.classList.remove('done'); }, 1600);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, function(){});
      } else {
        var ta = document.createElement('textarea');
        ta.value = text; document.body.appendChild(ta); ta.select();
        try { document.execCommand('copy'); done(); } catch(e){}
        document.body.removeChild(ta);
      }
    });
    box.appendChild(btn);
  });

  /* ---- Mermaid ---- */
  var mermaidNodes = document.querySelectorAll('.mermaid');
  var mermaidReady = false;
  function renderMermaid(force){
    if (!mermaidNodes.length) return;
    if (!window.mermaid) return;
    Array.prototype.forEach.call(mermaidNodes, function(el){
      if (force) { el.removeAttribute('data-processed'); el.innerHTML = el.dataset.code || ''; }
      else if (el.dataset.code && !el.innerHTML.trim()) { el.innerHTML = el.dataset.code; }
    });
    try {
      window.mermaid.initialize({
        startOnLoad: false,
        securityLevel: 'loose',
        theme: 'default',
        themeVariables: { background: '#FFFFFF', primaryColor: '#E3F7F8', primaryTextColor: '#0E1B33',
            primaryBorderColor: '#14B8C2', lineColor: '#6B7894', secondaryColor: '#E3F7F8',
            tertiaryColor: '#F7F9FC' }
      });
      window.mermaid.run({ querySelector: '.mermaid' });
      mermaidReady = true;
    } catch(e) {}
  }
  if (mermaidNodes.length) {
    if (!window.mermaid) {
      var s = document.createElement('script');
      s.src = 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js';
      s.onload = function(){ renderMermaid(false); };
      document.head.appendChild(s);
    } else {
      renderMermaid(false);
    }
  }

  /* ---- 阅读进度 ---- */
  var bar = document.getElementById('progress');
  if (bar) {
    var onScroll = function(){
      var h = document.documentElement;
      var max = h.scrollHeight - h.clientHeight;
      bar.style.width = max > 0 ? (h.scrollTop / max * 100) + '%' : '0';
    };
    window.addEventListener('scroll', onScroll, {passive:true});
    onScroll();
  }

  /* ---- TOC 高亮 ---- */
  var tocLinks = Array.prototype.slice.call(document.querySelectorAll('.toc-link'));
  if (tocLinks.length) {
    var targets = tocLinks.map(function(a){
      return document.getElementById(decodeURIComponent(a.getAttribute('href').slice(1)));
    });
    var tick = function(){
      var pos = window.scrollY + 120;
      var idx = 0;
      for (var i = 0; i < targets.length; i++){
        if (targets[i] && targets[i].offsetTop <= pos) idx = i;
      }
      tocLinks.forEach(function(a, i){ a.classList.toggle('active', i === idx); });
    };
    window.addEventListener('scroll', tick, {passive:true});
    tick();
  }

  /* ---- 搜索 ---- */
  var input = document.getElementById('searchInput');
  var results = document.getElementById('searchResults');
  var INDEX = null, loading = false, sel = -1, shown = [];

  function loadIndex(cb){
    if (INDEX) { cb(INDEX); return; }
    if (loading) { return; }
    loading = true;
    var base = document.querySelector('script[src*="app.js"]').getAttribute('src').replace(/app\.js\?v=.*$/, '');
    fetch(base + 'search-index.json').then(function(r){ return r.json(); })
      .then(function(d){ INDEX = d; loading = false; cb(d); })
      .catch(function(){ loading = false; cb([]); });
  }

  function snippet(text, q){
    var i = text.toLowerCase().indexOf(q.toLowerCase());
    if (i < 0) return text.slice(0, 92);
    var s = Math.max(0, i - 34);
    return (s > 0 ? '…' : '') + text.slice(s, s + 110) + '…';
  }

  function search(q){
    q = q.trim().toLowerCase();
    if (!q) { results.classList.remove('on'); return; }
    loadIndex(function(docs){
      var hits = [];
      docs.forEach(function(d){
        var t = d.t.toLowerCase(), x = (d.x || '').toLowerCase();
        var score = 0;
        if (t.indexOf(q) === 0) score += 40;
        else if (t.indexOf(q) > -1) score += 22;
        var xi = x.indexOf(q);
        if (xi > -1) score += 6 + Math.max(0, 6 - xi / 400);
        if (score > 0) hits.push({d: d, s: score});
      });
      hits.sort(function(a,b){ return b.s - a.s; });
      shown = hits.slice(0, 18);
      sel = -1;
      if (!shown.length) {
        results.innerHTML = '<div class="sr-empty">没有匹配的内容</div>';
        results.classList.add('on');
        return;
      }
      // 当前页面所在目录层级（相对站点根）：首页在根目录 → 0 个 '../'，其余子目录页 → 1 个
      var dir = location.pathname.replace(/\/[^/]*$/, '');
      var levels = dir.split('/').filter(Boolean).length;
      var prefix = '../'.repeat(Math.max(0, levels - 1));
      results.innerHTML = shown.map(function(h){
        var d = h.d;
        return '<a class="sr-item" href="' + prefix + d.u + '">' +
          '<div class="sr-t"><span>' + d.n + '</span>' + esc(d.t) + '</div>' +
          '<div class="sr-m">' + esc(d.m) + '</div>' +
          '<div class="sr-x">' + esc(snippet(d.x || '', q)) + '</div></a>';
      }).join('');
      results.classList.add('on');
    });
  }

  function esc(s){
    return String(s).replace(/[&<>"]/g, function(c){
      return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c];
    });
  }

  if (input) {
    var timer;
    input.addEventListener('input', function(){
      clearTimeout(timer);
      var v = input.value;
      timer = setTimeout(function(){ search(v); }, 130);
    });
    input.addEventListener('keydown', function(e){
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        e.preventDefault();
        var items = results.querySelectorAll('.sr-item');
        if (!items.length) return;
        sel = e.key === 'ArrowDown' ? Math.min(sel + 1, items.length - 1) : Math.max(sel - 1, 0);
        Array.prototype.forEach.call(items, function(el, i){ el.classList.toggle('sel', i === sel); });
        items[sel].scrollIntoView({block:'nearest'});
      } else if (e.key === 'Enter') {
        var items2 = results.querySelectorAll('.sr-item');
        if (items2.length) { location.href = items2[Math.max(0, sel)].getAttribute('href'); }
      } else if (e.key === 'Escape') {
        input.value = ''; results.classList.remove('on'); input.blur();
      }
    });
    document.addEventListener('click', function(e){
      if (!results.contains(e.target) && e.target !== input) results.classList.remove('on');
    });
  }

  document.addEventListener('keydown', function(e){
    if (e.key === '/' && document.activeElement !== input &&
        !/input|textarea|select/i.test(document.activeElement.tagName)) {
      e.preventDefault();
      if (window.innerWidth <= 960) openSb();
      input.focus();
    }
  });
})();
