<script>
    import Logo from './components/Logo.svelte';
    import Toolbar from './components/Toolbar.svelte';
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

    .main {
        height: 200rem;
        width: 200rem;
    }
</style>

<div class="main">
    <Logo />
    <div class="flex h-screen flex-col justify-center">
        <Toolbar />
    </div>
</div>
