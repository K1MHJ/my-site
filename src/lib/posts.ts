// src/lib/posts.ts
import type { MarkdownInstance } from 'astro';

export type PostFrontmatter = {
  id: string;
  slug: string;
  lang?: string;
  title: string;
  summary?: string;
  createdAt: string;
  updatedAt?: string;
  tags?: string[];
  category?: string;
  heroImage?: string;
  draft?: boolean;
  series?: string;
  orderInSeries?: number;
};

export type Post = PostFrontmatter & {
  url: string; // /posts/slug/
};

// 루트 content/posts/**/index.ko.md 를 모두 읽어온다
const postModules = import.meta.glob<MarkdownInstance<PostFrontmatter>>(
  '../../content/posts/**/index.ko.md',
  { eager: true }
);

export function getAllPosts(): Post[] {
  const posts: Post[] = [];

  for (const mod of Object.values(postModules)) {
    const fm = mod.frontmatter;
    if (fm.draft) continue;

    posts.push({
      ...fm,
      url: `/posts/${fm.slug}/`,
    });
  }

  posts.sort((a, b) => {
    const da = new Date(a.createdAt).getTime();
    const db = new Date(b.createdAt).getTime();
    return db - da;
  });

  return posts;
}

export function getPostBySlug(slug: string) {
  for (const mod of Object.values(postModules)) {
    if (mod.frontmatter.slug === slug) {
      return {
        frontmatter: mod.frontmatter,
        Content: mod.default,
      };
    }
  }
  return undefined;
}
