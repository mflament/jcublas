package org.yah.tools.cuda.kernel;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.util.Map;
import java.util.Objects;

public class ModuleInvocationHandler implements InvocationHandler {
    private final Map<String, KernelInvocationHandler> handlers;

    public ModuleInvocationHandler(Map<String, KernelInvocationHandler> handlers) {
        this.handlers = Map.copyOf(Objects.requireNonNull(handlers, "handlers is null"));
    }

    @Override
    public Object invoke(Object proxy, Method method, Object[] args) {
        if (method.getName().equals("equals") && method.getParameterCount() == 1)
            return Objects.equals(proxy, args[0]);
        if (method.getName().equals("hashCode") && method.getParameterCount() == 0)
            return proxy.hashCode();
        if (method.getName().equals("toString") && method.getParameterCount() == 0)
            return proxy.toString();

        KernelInvocationHandler invocationHandler = handlers.get(method.getName());
        if (invocationHandler == null)
            throw new IllegalStateException("Unhandled method " + method);
        invocationHandler.invoke(args);
        return null;
    }
}
