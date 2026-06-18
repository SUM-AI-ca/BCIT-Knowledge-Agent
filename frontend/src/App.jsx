import { lazy, Suspense } from "react";

// Route-level code splitting: the blog deep-dive and the chat app are loaded
// as separate chunks, so a /chat visitor never downloads the blog page (and
// vice versa). Pathname routing — still no router dependency.
const Blog = lazy(() => import("./Blog"));
const Chat = lazy(() => import("./Chat"));

export default function App() {
  const path = window.location.pathname;
  const Page = path === "/chat" || path.startsWith("/chat/") ? Chat : Blog;
  return (
    <Suspense fallback={<div style={{ minHeight: "100vh" }} aria-hidden="true" />}>
      <Page />
    </Suspense>
  );
}
