---
layout: default
title: Open Source
permalink: /opensource/
---

# Open Source

<ul>
  {% for oss in site.opensource %}
    <li>
      <a href="{{ oss.url }}">{{ oss.title }}</a>
      <span style="color:#888;">({{ oss.date | date: '%Y-%m-%d' }})</span>
    </li>
  {% endfor %}
</ul> 