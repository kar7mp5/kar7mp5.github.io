# Min-Sup Kim's Blog

Welcome to the source code for Min-Sup Kim's personal blog and portfolio site!

## How to Write a Blog Post

1. **Create a new post file**
   - Go to the `_posts/` directory.
   - Create a new file named `YYYY-MM-DD-title.md` (e.g., `2024-07-18-my-first-post.md`).

2. **Add front matter**
   - At the top of your file, add:
     ```
     ---
     layout: default
     title: "Your Post Title"
     categories: [dev]   # or [life], [project], etc.
     tags: [tag1, tag2]  # optional
     ---
     ```

3. **Write your content**
   - Use Markdown for formatting (headings, lists, code, etc.).
   - Example:
     ```markdown
     # My First Post

     This is my first blog post!

     ## Section

     - Bullet point
     - Another point

     ```python
     print("Hello, world!")
     ```
     ```

4. **Add images**
   - Place your image in the `assets/img/` folder.
   - Reference it in your post:
     ```markdown
     ![Description](/assets/img/myimage.jpg)
     ```

5. **MathJax (Math formulas)**
   - Inline: `$E = mc^2$`
   - Block: `$$a^2 + b^2 = c^2$$`

6. **Preview locally**
   - Run `jekyll serve --port 4001` and visit `http://localhost:4001`.

7. **Commit and push**
   - Save your file, commit, and push to GitHub. If using GitHub Pages, your post will be published automatically.

---

## Useful Folders
- `_posts/` : Blog posts
- `assets/img/` : Images
- `_includes/sidebar.html` : Sidebar menu
- `home.md`, `biography.md`, etc.: Main pages

## Tips
- To add a new page, create a `.md` file with front matter and link it in the sidebar.
- To customize the sidebar, edit `_includes/sidebar.html`.
- To change your profile image, replace `assets/img/profile.jpg`.

---
For more details, see `HANDOFF.md` or contact the maintainer.