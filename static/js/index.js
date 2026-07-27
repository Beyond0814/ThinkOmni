document.addEventListener("DOMContentLoaded", () => {
  if (window.lucide) {
    window.lucide.createIcons();
  }

  document.querySelectorAll(".navbar-burger").forEach((burger) => {
    burger.addEventListener("click", () => {
      const menu = document.getElementById(burger.dataset.target);
      const active = burger.classList.toggle("is-active");
      menu?.classList.toggle("is-active", active);
      burger.setAttribute("aria-expanded", String(active));
    });
  });

  document.querySelectorAll(".navbar-menu a").forEach((link) => {
    link.addEventListener("click", () => {
      document.querySelectorAll(".navbar-burger, .navbar-menu").forEach((element) => {
        element.classList.remove("is-active");
      });
      document.querySelectorAll(".navbar-burger").forEach((burger) => {
        burger.setAttribute("aria-expanded", "false");
      });
    });
  });

  const copyButton = document.querySelector(".copy-button");
  copyButton?.addEventListener("click", async () => {
    const target = document.getElementById(copyButton.dataset.copyTarget);
    if (!target) return;
    await navigator.clipboard.writeText(target.innerText);
    const label = copyButton.querySelector("span");
    label.textContent = "Copied";
    window.setTimeout(() => { label.textContent = "Copy"; }, 1600);
  });
});
