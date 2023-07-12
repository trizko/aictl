<script>
    import Logo from './components/Logo.svelte';
</script>

<style type="text/postcss">
    @tailwind base;
    @tailwind components;
    @tailwind utilities;

    :global(body) {
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
    <Logo />
</div>
