(function () {
  "use strict";

  function slugify(text, index) {
    var slug = text
      .toLowerCase()
      .trim()
      .replace(/\s+/g, "-")
      .replace(/[^a-z0-9\-_]/g, "");

    if (!slug) {
      slug = "section-" + (index + 1);
    }

    return slug;
  }

  function buildToc() {
    var tocNav = document.querySelector(".toc");
    if (!tocNav) {
      return;
    }

    var content = document.querySelector(".post-content");
    if (!content) {
      tocNav.style.display = "none";
      return;
    }

    var headings = content.querySelectorAll("h2, h3, h4");
    if (!headings.length) {
      tocNav.style.display = "none";
      return;
    }

    var listRoot = tocNav.querySelector(".toc-list");
    if (!listRoot) {
      tocNav.style.display = "none";
      return;
    }

    var currentLevel = 2;
    var currentList = listRoot;
    var listStack = [listRoot];

    headings.forEach(function (heading, index) {
      var level = parseInt(heading.tagName.substring(1), 10);
      if (!heading.id) {
        heading.id = slugify(heading.textContent || "", index);
      }

      while (level > currentLevel) {
        var lastItem = currentList.lastElementChild;
        if (!lastItem) {
          break;
        }
        var nestedList = document.createElement("ol");
        lastItem.appendChild(nestedList);
        listStack.push(nestedList);
        currentList = nestedList;
        currentLevel += 1;
      }

      while (level < currentLevel && listStack.length > 1) {
        listStack.pop();
        currentList = listStack[listStack.length - 1];
        currentLevel -= 1;
      }

      var listItem = document.createElement("li");
      var link = document.createElement("a");
      link.href = "#" + heading.id;
      link.textContent = heading.textContent || "";
      listItem.appendChild(link);
      currentList.appendChild(listItem);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", buildToc);
  } else {
    buildToc();
  }
})();
