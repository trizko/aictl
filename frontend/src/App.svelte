<script>
    import Toolbar from './components/Toolbar.svelte';
    import ThemeToggle from './components/ThemeToggle.svelte';
</script>

<style type="text/postcss">
    @tailwind base;
    @tailwind components;
    @tailwind utilities;

    :global(body) {
        margin: 0;
        padding: 0;
        background: #c9c9c9ea;
        background-size: 150px 150px;
        background-image:
            linear-gradient(to right, rgba(0, 0, 0, 0.2) 1px, transparent 1px),
            linear-gradient(to bottom, rgba(0, 0, 0, 0.2) 1px, transparent 1px);
    }

    :global(body.dark) {
        margin: 0;
        padding: 0;
        background: #000000ea;
        background-size: 150px 150px;
        background-image:
            linear-gradient(to right, rgb(255, 255, 255, 0.2) 1px, transparent 1px),
            linear-gradient(to bottom, rgb(255, 255, 255, 0.2) 1px, transparent 1px);
    }
</style>

<div>
    <div class="ml-2 mt-2 fixed">
        <Toolbar />
    </div>
</div>
<ThemeToggle />
