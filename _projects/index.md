---
layout: default
title: Projects
permalink: /projects/
---

# Projects

<ul>
  {% for project in site.projects %}
    <li>
      <a href="{{ project.url }}">{{ project.title }}</a>
      <span style="color:#888;">({{ project.date | date: '%Y-%m-%d' }})</span>
    </li>
  {% endfor %}
</ul> 