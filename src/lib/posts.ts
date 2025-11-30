// src/lib/posts.ts
import type { MarkdownInstance } from 'astro';

type PostFrontmatter = {
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

// content/posts/**/index.ko.md 를 전부 읽어온다.
const postModules = import.meta.glob<MarkdownInstance<PostFrontmatter>>(
  '../../content/posts/**/index.ko.md',
  { eager: true }
);

/** 모든 글 가져오기 (draft 제외, 최신순 정렬) */
export function getAllPosts(): Post[] {
  const posts: Post[] = [];

  for (const mod of Object.values(postModules)) {
    const fm = mod.frontmatter;

    // draft 글은 리스트에서 제외
    if (fm.draft) continue;

    posts.push({
      ...fm,
      url: `/posts/${fm.slug}/`,
    });
  }

  // createdAt 기준으로 내림차순 정렬
  posts.sort((a, b) => {
    const da = new Date(a.createdAt).getTime();
    const db = new Date(b.createdAt).getTime();
    return db - da;
  });

  return posts;
}

/** slug로 글 하나 찾기 (본문 컴포넌트 포함) */
export function getPostBySlug(slug: string): {
  frontmatter: PostFrontmatter;
  Content: MarkdownInstance<PostFrontmatter>['default'];
} | undefined {
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
