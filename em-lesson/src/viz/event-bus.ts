type Listener = (...args: unknown[]) => void;

class EventBus {
  private listeners: Map<string, Listener[]> = new Map();
  on(event: string, fn: Listener) {
    const arr = this.listeners.get(event);
    if (arr) {
      arr.push(fn);
    } else {
      this.listeners.set(event, [fn]);
    }
  }
  emit(event: string, ...args: unknown[]) {
    this.listeners.get(event)?.forEach(fn => fn(...args));
  }
}

export const emBus = new EventBus();
