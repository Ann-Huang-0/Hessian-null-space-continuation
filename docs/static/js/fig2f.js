// Interactive Fig 2F: two MDS maps of the same 131 networks (1 - CKA, relative weight distance).
// Hovering or clicking picks the nearest network; it is ringed in both maps and its task video plays.
(function () {
  var root = document.getElementById('f2f');
  if (!root) return;
  var BASE = './static/fig2f/';
  var NS = 'http://www.w3.org/2000/svg';
  var COL = { anchor: '#08886D', und: '#A46BBD', cka: '#1B3F82', dsa: '#C28412', seed: '#808080' };
  var FAM = { anchor: 'anchor', und: 'undirected', cka: 'CKA-steered', dsa: 'DSA-steered', seed: 'trained seed' };
  var ORDER = { seed: 0, cka: 1, dsa: 1, und: 2, anchor: 3 };   // paper draw order: seeds under, undirected over the steered arms
  var V = 300, L = 22, B = 22, P = 4, S = V - L - P;           // viewBox, left/bottom axis gutters, plot side
  var PICK = 16;                                               // pick radius, in viewBox units
  var video = root.querySelector('video'), label = root.querySelector('.f2f-label');
  var nets = [], maps = [], selected = null, hovered = null;

  function el(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function radius(n) {                                          // matplotlib s (pt^2) -> radius, as in the paper panel
    var s = n.family === 'seed' ? 28 : 20 + 20 * n.frac;
    return 0.95 * Math.sqrt(s);
  }

  function buildMap(svg, key) {
    var xs = nets.map(function (n) { return n[key][0]; }), ys = nets.map(function (n) { return n[key][1]; });
    var cx = (Math.min.apply(null, xs) + Math.max.apply(null, xs)) / 2, cy = (Math.min.apply(null, ys) + Math.max.apply(null, ys)) / 2;
    var h = 1.06 * Math.max(Math.max.apply(null, xs) - Math.min.apply(null, xs), Math.max.apply(null, ys) - Math.min.apply(null, ys)) / 2;
    var px = function (x) { return L + (x - cx + h) / (2 * h) * S; }, py = function (y) { return P + S - (y - cy + h) / (2 * h) * S; };
    svg.setAttribute('viewBox', '0 0 ' + V + ' ' + V);
    el('path', { d: 'M' + L + ' ' + P + 'V' + (P + S) + 'H' + (L + S), fill: 'none', stroke: '#3B3B38', 'stroke-width': 1.3 }, svg);
    el('text', { x: L + S / 2, y: V - 5, 'text-anchor': 'middle' }, svg).textContent = 'MDS 1';
    el('text', { x: 14, y: P + S / 2, 'text-anchor': 'middle', transform: 'rotate(-90 14 ' + (P + S / 2) + ')' }, svg).textContent = 'MDS 2';
    var dots = el('g', {}, svg), rings = el('g', { 'pointer-events': 'none' }, svg);
    var pos = nets.map(function (n) { return [px(n[key][0]), py(n[key][1])]; });
    nets.map(function (n, i) { return i; })
      .sort(function (a, b) { return ORDER[nets[a].family] - ORDER[nets[b].family] || a - b; })
      .forEach(function (i) {
        var n = nets[i], p = pos[i];
        if (n.family === 'anchor') {
          el('rect', { x: p[0] - 5.5, y: p[1] - 5.5, width: 11, height: 11, fill: 'none', stroke: COL.anchor, 'stroke-width': 2.4 }, dots);
        } else {
          el('circle', { cx: p[0], cy: p[1], r: radius(n), fill: COL[n.family],
                         'fill-opacity': n.family === 'seed' ? 0.9 : 0.45 + 0.55 * n.frac }, dots);
        }
      });
    var hov = el('circle', { r: 0, fill: 'none', stroke: '#9a9a9a', 'stroke-width': 1.6, visibility: 'hidden' }, rings);
    var sel = el('circle', { r: 0, fill: 'none', stroke: '#1d1d1b', 'stroke-width': 2.4, visibility: 'hidden' }, rings);
    var map = { svg: svg, pos: pos, hov: hov, sel: sel };

    function nearest(evt) {
      var m = svg.getScreenCTM();
      if (!m) return null;
      var pt = svg.createSVGPoint(); pt.x = evt.clientX; pt.y = evt.clientY;
      var q = pt.matrixTransform(m.inverse()), best = null, bd = PICK * PICK;
      pos.forEach(function (p, i) {
        var d = (p[0] - q.x) * (p[0] - q.x) + (p[1] - q.y) * (p[1] - q.y);
        if (d < bd) { bd = d; best = i; }
      });
      return best;
    }
    svg.addEventListener('pointermove', function (evt) { setHover(nearest(evt)); });
    svg.addEventListener('pointerleave', function () { setHover(null); });
    svg.addEventListener('click', function (evt) { var i = nearest(evt); if (i !== null) select(i); });
    return map;
  }

  function place(ring, map, i, pad) {
    if (i === null) { ring.setAttribute('visibility', 'hidden'); return; }
    var n = nets[i];
    ring.setAttribute('cx', map.pos[i][0]); ring.setAttribute('cy', map.pos[i][1]);
    ring.setAttribute('r', (n.family === 'anchor' ? 8.5 : radius(n)) + pad);
    ring.setAttribute('visibility', 'visible');
  }
  function setHover(i) {
    hovered = i;
    maps.forEach(function (m) {
      place(m.hov, m, i === selected ? null : i, 3);
      m.svg.classList.toggle('is-over', i !== null);
    });
  }

  function describe(n) {
    var name = n.family === 'anchor' ? 'anchor'
      : n.family === 'seed' ? 'independently trained network'
      : FAM[n.family] + ' walk ' + n.walk_no + ', step ' + n.step;
    var fp = n.n_fp === null ? '' : n.n_fp + ' stable fixed point' + (n.n_fp === 1 ? '' : 's');
    var dist = n.family === 'anchor' ? ''
      : '1 &minus; CKA = ' + n.dcka.toFixed(2) + ' &middot; &Vert;&Delta;&theta;&Vert;/&Vert;&theta;<sub>0</sub>&Vert; = ' + n.dw.toFixed(2);
    var drift = n.correct === false ? '<br><span class="f2f-dist">drifts off the target in the long gaps between pulses</span>' : '';
    return '<span style="color:' + COL[n.family] + '">' + name + '</span><br>' + fp +
      (dist ? '<br><span class="f2f-dist">' + dist + '</span>' : '') + drift;
  }

  function select(i) {
    selected = i;
    maps.forEach(function (m) { place(m.sel, m, i, 3.5); place(m.hov, m, null, 0); });
    var n = nets[i], id = String(i).padStart(3, '0');
    label.innerHTML = describe(n);
    if (n.n_fp !== null) {
      video.poster = BASE + 'net_' + id + '.webp';
      video.src = BASE + 'net_' + id + '.mp4';
      var p = video.play(); if (p && p.catch) p.catch(function () {});
    }
  }

  fetch(BASE + 'networks.json').then(function (r) { return r.json(); }).then(function (d) {
    nets = d.networks;
    maps = [buildMap(root.querySelector('svg[data-key="cka"]'), 'cka'), buildMap(root.querySelector('svg[data-key="w"]'), 'w')];
    select(0);
  });
})();
