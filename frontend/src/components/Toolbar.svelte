<script>
    import Logo from "./Logo.svelte";
    import Prompt from "./Prompt.svelte";
</script>

<style type='text/postcss'>
    @tailwind base;
    @tailwind components;
    @tailwind utilities;
</style>

<div class="flex h-10">
    <Logo />
    <Prompt />
</div>