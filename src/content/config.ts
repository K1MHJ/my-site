import { defineCollection, z } from "astro:content";

const blog = defineCollection({
  type: "content",
  schema: z.object({
    title: z.string(),
    date: z.string().transform((s) => new Date(s)), // "2025-11-09" 형식
    description: z.string().optional(),
    tags: z.array(z.string()).default([]),
    category: z.string().optional(),
    thumbnail: z.string().optional(), // 목록 썸네일
    draft: z.boolean().default(false),
  }),
});

export const collections = { blog };
