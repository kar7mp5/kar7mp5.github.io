# Project Handoff Guide

## 1. Project Structure

- `_layouts/` : HTML layout templates (default.html is the main layout)
- `_includes/` : Reusable HTML snippets (sidebar, footer, etc.)
- `assets/css/` : CSS files (jemdoc.css for main style)
- `assets/img/` : Images (profile, etc.)
- `home.md`, `biography.md`, `blog.html`, etc.: Main content pages
- `categories/`, `projects.md`, `opensource.md`, etc.: Category and project pages
- `_posts/` : Blog posts (Markdown files)

## 2. How to Add/Edit Pages

- **Add a new page:**
  1. Create a new `.md` or `.html` file in the root or relevant folder.
  2. Add YAML front matter at the top:
     ```
     ---
     layout: default
     title: Your Page Title
     ---
     ```
  3. Write your content in Markdown or HTML below the front matter.
  4. Add a link to the new page in the sidebar if needed (`_includes/sidebar.html`).

- **Edit an existing page:**
  - Open the relevant `.md` or `.html` file and modify the content.

## 3. How to Add a Blog Post

- Go to the `_posts/` directory.
- Create a new file named `YYYY-MM-DD-title.md` (e.g., `2024-07-18-my-first-post.md`).
- Add the following front matter:
  ```
  ---
  layout: default
  title: "My First Post"
  categories: [dev]
  ---
  ```
- Write your post content in Markdown below the front matter.
- Save and commit the file.

## 4. Customization Tips

- **Sidebar:** Edit `_includes/sidebar.html` to change menu structure.
- **Profile Image:** Place your image in `assets/img/profile.jpg` and edit `home.md` if needed.
- **Styling:** Edit `assets/css/jemdoc.css` for custom styles.
- **MathJax:** Math formulas can be written using `$...$` for inline or `$$...$$` for block math.

## 5. Local Development

- Run `jekyll serve --port 4001` to preview your site at `http://localhost:4001`.
- If port 4001 is busy, use another port (e.g., `--port 4002`).

## 6. Deployment

- Push your changes to GitHub. If using GitHub Pages, your site will be automatically deployed.

---
For any further questions, check the README.md or contact the previous maintainer. 