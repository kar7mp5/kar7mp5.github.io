// Add copy button to all code blocks
window.addEventListener('DOMContentLoaded', function () {
  document.querySelectorAll('pre.highlight').forEach(function (pre) {
    // Create button
    var btn = document.createElement('button');
    btn.innerText = 'Copy';
    btn.className = 'copy-btn';
    btn.style.position = 'absolute';
    btn.style.top = '8px';
    btn.style.right = '8px';
    btn.style.padding = '2px 8px';
    btn.style.fontSize = '0.9em';
    btn.style.background = '#f6f8fa';
    btn.style.border = '1px solid #ccc';
    btn.style.borderRadius = '4px';
    btn.style.cursor = 'pointer';
    btn.style.zIndex = '10';
    btn.addEventListener('click', function () {
      var code = pre.querySelector('code');
      if (code) {
        navigator.clipboard.writeText(code.innerText).then(function () {
          btn.innerText = 'Copied!';
          setTimeout(function () { btn.innerText = 'Copy'; }, 1200);
        });
      }
    });
    // Make pre relative for button positioning
    pre.style.position = 'relative';
    pre.appendChild(btn);
  });
});

// Optional: style for copy button (if not in CSS)
var style = document.createElement('style');
style.innerHTML = '.copy-btn:active { background: #e1e4e8; }';
document.head.appendChild(style); 