---
layout: default
title: Portfolio
permalink: /portfolio/
---

# Portfolio

<ul>
  {% for item in site.portfolio %}
    <li>
      <a href="{{ item.url }}">{{ item.title }}</a>
      <span style="color:#888;">({{ item.date | date: '%Y-%m-%d' }})</span>
    </li>
  {% endfor %}
</ul> 