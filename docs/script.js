// ELOEdge -- live ratings tabs + stat counters
(function () {
  let RATINGS = null;
  let activeSport = 'NBA';

  async function loadRatings() {
    try {
      const res = await fetch('data/ratings.json', { cache: 'no-store' });
      RATINGS = await res.json();
      renderStats();
      renderRatings(activeSport);
    } catch (e) {
      console.error('failed to load ratings', e);
      const wrap = document.querySelector('.ratings-table-wrap');
      if (wrap) wrap.innerHTML = '<div style="padding:24px;color:#8d97a8;font-family:var(--mono)">Ratings unavailable. Check data/ratings.json.</div>';
    }
  }

  function renderStats() {
    document.querySelectorAll('.stat-num').forEach(el => {
      const sport = el.dataset.sport;
      const n = (RATINGS[sport]?.all?.length) || 0;
      animateNum(el, n);
    });
  }

  function animateNum(el, target) {
    let cur = 0;
    const step = Math.max(1, Math.ceil(target / 24));
    const tick = () => {
      cur = Math.min(target, cur + step);
      el.textContent = cur;
      if (cur < target) requestAnimationFrame(tick);
    };
    tick();
  }

  function renderRatings(sport) {
    if (!RATINGS) return;
    const data = RATINGS[sport];
    if (!data) return;

    const meta = document.getElementById('ratings-meta');
    if (meta) {
      const m = data.meta || {};
      const parts = [];
      if (m.season_label) parts.push('Season ' + m.season_label);
      if (m.trained_games) parts.push(m.trained_games.toLocaleString() + ' games trained');
      if (m.saved_at) parts.push('Updated ' + m.saved_at.split(' ')[0]);
      meta.textContent = parts.join('  ·  ');
    }

    const tbody = document.querySelector('#ratings-table tbody');
    tbody.innerHTML = '';
    const teams = data.all || [];
    // Fixed ±400 Elo scale: +400 ≈ 91% win expectation vs 1500 opponent.
    // Bar fills half of the track (50%), so max pct per side = 50.
    const SCALE = 400;

    teams.forEach((t, i) => {
      const tr = document.createElement('tr');
      const delta = t.rating - 1500;
      const pct = Math.min(Math.abs(delta) / SCALE, 1) * 50;
      const side = delta >= 0 ? 'pos' : 'neg';
      const deltaRound = Math.round(delta);
      const deltaLabel = (deltaRound >= 0 ? '+' : '') + deltaRound;
      tr.innerHTML = `
        <td class="rank">${i + 1}</td>
        <td class="team">${escapeHtml(t.team)}</td>
        <td class="num">${t.rating.toFixed(0)}</td>
        <td class="bar-cell">
          <span class="bar-track" role="img" aria-label="${deltaLabel} Elo vs 1500">
            <span class="bar-fill ${side}" style="width:${pct.toFixed(1)}%"></span>
          </span>
          <span class="bar-delta ${side}">${deltaLabel}</span>
        </td>
      `;
      tbody.appendChild(tr);
    });
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[c]));
  }

  function bindTabs() {
    document.querySelectorAll('.tab').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        activeSport = btn.dataset.tab;
        renderRatings(activeSport);
      });
    });
  }

  function smoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(a => {
      a.addEventListener('click', (e) => {
        const id = a.getAttribute('href').slice(1);
        const target = document.getElementById(id);
        if (target) {
          e.preventDefault();
          target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      });
    });
  }

  document.addEventListener('DOMContentLoaded', () => {
    bindTabs();
    smoothScroll();
    loadRatings();
  });
})();
